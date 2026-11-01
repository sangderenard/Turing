"""Python thread syntax is replaced before executable graph/SSA planning."""
import ast

from tests.test_python_two_threads_shared_location import SOURCE, CONTRACT
from tests.test_source_ssa_synchronization import compile_source
from src.compiler.identity_concordance import concordance_report
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def graph_from_source(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source, resolve_unresolved_parents=True,
                         pursuit_roots=("root",), parent_include=CONTRACT)
    return graph


def test_ingestion_replaces_thread_callable_nodes_and_keeps_only_operand_ports():
    graph = graph_from_source(SOURCE)
    operations = [(node, data) for node, data in graph.G.nodes(data=True)
                  if (data.get("attributes") or {}).get("dispatch_operation", {}).get("operation", "").startswith("thread_")]
    assert len(operations) == 6
    expected = {"thread_create": "ThreadCreate", "thread_start": "ThreadStart", "thread_join": "ThreadJoin"}
    for node, data in operations:
        operation = data["attributes"]["dispatch_operation"]["operation"]
        assert data["type"] == expected[operation]
        assert all(edge.get("role") != "func" for *_ends, edge in graph.G.in_edges(node, data=True))
    assert not any(data.get("type") == "Call" and
                   (data.get("attributes") or {}).get("extraction_identity") == "threading.Thread"
                   for _, data in graph.G.nodes(data=True))


def test_nested_thread_constructor_is_dataflow_not_an_extra_control_scope():
    graph = graph_from_source("""\
import threading
def root():
    def worker():
        return 1
    handles = []
    i = 0
    while i < 2:
        handles.append(threading.Thread(target=worker))
        i += 1
    return 1
""")
    creates = [(node, data) for node, data in graph.G.nodes(data=True) if data.get("type") == "ThreadCreate"]
    assert len(creates) == 1  # one source site, dynamically activated twice
    node, data = creates[0]
    scopes = data["attributes"]["dispatch_operation"]["control_scopes"]
    assert any(kind == "While" and arm == "body" for kind, _line, arm in scopes)
    assert all(kind != "Call" for kind, _line, _arm in scopes)
    assert any(isinstance(graph.G.nodes[target].get("expr_obj"), ast.Call)
               for _source, target in graph.G.out_edges(node))


def test_source_thread_semantics_survive_to_ssa_without_python_callee_inputs():
    module = compile_source(SOURCE)
    operations = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs
                  if i.attributes.get("dispatch_operation", "").startswith("thread_")]
    assert [i.op for i in operations] == ["ThreadCreate", "ThreadCreate", "ThreadStart", "ThreadStart", "ThreadJoin", "ThreadJoin"]
    assert [i.res.dtype for i in operations] == ["opaque_ref", "opaque_ref", "none", "none", "none", "none"]
    assert all(i.attributes["effectful"] and i.attributes["ordered_effect"] for i in operations)
    assert not any(str(name).startswith("turing_dispatch_")
                   for f in module.functions.values() for name, _value in f.metadata.get("parameter_names", ()))
    assert "0 finding(s)" in concordance_report(module)


def test_condition_and_event_calls_are_replaced_in_source_graph_and_ssa():
    source = '''\
import threading
def root():
    condition = threading.Condition()
    event = threading.Event()
    condition.acquire()
    condition.notify_all()
    condition.release()
    event.set()
    event.clear()
    return 1
'''
    expected = ["ConditionCreate", "EventCreate", "ConditionAcquire",
                "ConditionNotifyAll", "ConditionRelease", "EventSet", "EventClear"]
    from src.compiler.thread_operations import PROCESS_RESULT_DTYPES
    graph = graph_from_source(source)
    assert sorted(data["type"] for _, data in graph.G.nodes(data=True)
                  if data.get("type") in PROCESS_RESULT_DTYPES) == sorted(expected)
    module = compile_source(source)
    operations = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs
                  if i.op in PROCESS_RESULT_DTYPES]
    assert [i.op for i in operations] == ["EventCreate", "EventSet", "EventClear"]
    assert all(i.attributes.get("source_process_cell") is not None for i in operations)
    sync = [i for f in module.functions.values() for b in f.blocks.values() for i in b.instrs
            if i.attributes.get("synchronization_receipt")]
    assert {i.op for i in sync} == {"MutexCreate", "WorkerToken", "WaitQueueMutex", "MutexAcquire",
                                  "MutexAcquired", "MutexToken", "MutexRelease", "WaiterSelect", "LatchSignal"}
    assert not any(i.op.startswith("Condition") for f in module.functions.values()
                   for b in f.blocks.values() for i in b.instrs)
    assert "0 finding(s)" in concordance_report(module)
