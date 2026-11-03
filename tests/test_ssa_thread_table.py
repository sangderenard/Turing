"""Worker tables built by the public Python-source compiler entry."""
import pytest

from src.compiler.concordance_declarations import SOURCE_CLOSURE_CAPTURE, NAME_BINDING
from src.compiler.identity_concordance import concordance_report
from tests.test_python_two_threads_shared_location import SOURCE
from tests.test_source_ssa_synchronization import compile_source


@pytest.fixture(scope="module")
def shared_module():
    return compile_source(SOURCE)


def test_two_handles_share_one_actual_ssa_body_with_distinct_context_rows(shared_module):
    module = shared_module
    table = module.thread_table
    assert table.book is module.metadata["identity_book"]
    assert len(table.bodies) == len(table.ssa_bodies) == 1
    assert len(table.activations) == len(table.ssa_activations) == len(table.contexts) == 2
    assert len({source.row[0] for source in table.activations}) == 2
    body, = table.bodies
    worker, = table.worker_functions(body)
    assert worker is module.functions[worker.name]
    assert worker.metadata["source_function_reference"] == table.bodies[body][0]
    instructions = [i for block in worker.blocks.values() for i in block.instrs]
    assert any(i.op == "Load" for i in instructions)
    assert any(i.op == "Store" for i in instructions)
    # The real body reaches its arithmetic region through the ordinary linker.
    calls = [i for i in instructions if i.op == "Call" and i.attributes.get("callee") in module.functions]
    assert any(i.op == "Add" for call in calls for b in module.functions[call.attributes["callee"]].blocks.values()
               for i in b.instrs)
    assert all(value.dtype is not None for value in module.functions["root__root"].args)
    assert all(value.accounting.get("compiler_frame_sequence_id") == 1
               for value in module.functions["root__root"].args)
    assert "0 finding(s)" in concordance_report(module)


def test_capture_contexts_keep_exact_parent_storage_and_all_sequence_members(shared_module):
    module = shared_module
    table = module.thread_table
    book = table.book
    contexts = list(table.contexts.items())
    assert contexts[0][0].row[1] == contexts[1][0].row[1]  # same closure binding, distinct creates
    for cell, (actual, formal, actual_members, formal_members, status) in contexts:
        capture = cell.row[1]
        assert capture.page == SOURCE_CLOSURE_CAPTURE
        _, parent_binding, input_binding = book.page(capture.page).cell(capture.row, capture.column)
        assert actual == book.page(NAME_BINDING).cell(parent_binding.row, parent_binding.column).value_id
        assert formal == book.page(NAME_BINDING).cell(input_binding.row, input_binding.column).value_id
        assert status == "live"
        assert any((actual, ("column", 0)) in roles for _, roles in actual_members)
        assert any((formal, ("length_address_id",)) in roles for _, roles in formal_members)
        assert any((formal, ("capacity_value_id",)) in roles for _, roles in formal_members)
    assert contexts[0][1][:4] == contexts[1][1][:4]
    root = module.functions["root__root"]
    creates = [i for b in root.blocks.values() for i in b.instrs if i.op == "ThreadCreate"]
    actual = contexts[0][1][0]
    assert all(i.attributes["capture_value_ids"] == (actual,) for i in creates)
    assert all([a.id for a in i.args] == [actual] for i in creates)
    for fact in table.activations.values():
        assert len(fact[3]) == len(fact[4]) == 1  # start and completion dependency


def test_worker_argument_and_branch_body_are_retained_without_an_ordinary_parent_call():
    module = compile_source('''\
import threading
def worker(value: int):
    if value > 0:
        return value + 1
    return value - 1
def root():
    thread = threading.Thread(target=worker, args=(7,))
    thread.start()
    thread.join()
    return 1
''')
    table = module.thread_table
    assert len(table.bodies) == len(table.activations) == len(table.ssa_activations) == 1
    body, = table.bodies
    worker, = table.worker_functions(body)
    assert any(i.op == "CondBr" for b in worker.blocks.values() for i in b.instrs)
    assert any(i.attributes.get("worker_argument_pack_expanded") and len(i.attributes.get("worker_arguments", ())) == 1 for b in module.functions["root__root"].blocks.values()
               for i in b.instrs if i.op == "ThreadCreate")
    assert not any(i.op == "Call" and i.attributes.get("callee") == worker.name
                   for b in module.functions["root__root"].blocks.values() for i in b.instrs)
    assert "0 finding(s)" in concordance_report(module)


def test_unrelated_unused_receiver_has_a_recorded_empty_physical_context():
    module = compile_source('''\
import threading
class Other:
    def wait(self):
        return 3
def root():
    unrelated = Other()
    def worker():
        unrelated.wait()
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return 1
''')
    assert len(module.thread_table.contexts) == 1
    assert next(iter(module.thread_table.contexts.values()))[-1] == "unused"
    assert not any(i.op in {"ControlBarrier", "MemoryBarrier"}
                   for f in module.functions.values() for b in f.blocks.values() for i in b.instrs)
    assert "0 finding(s)" in concordance_report(module)


def test_create_in_loop_is_one_factory_row_with_its_authored_control_scope():
    module = compile_source('''\
import threading
def worker():
    return 2
def root():
    i = 0
    while i < 3:
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        i += 1
    return i
''')
    table = module.thread_table
    assert len(table.activations) == len(table.ssa_activations) == 1
    activation = next(iter(table.activations.values()))
    assert any(kind == "While" and arm == "body" for kind, _line, arm in activation[5])
    assert len(activation[3]) == len(activation[4]) == 1
    assert "0 finding(s)" in concordance_report(module)


@pytest.mark.parametrize("initializer", ["[9]", "9"])
def test_read_only_capture_is_bound_to_its_actual_source_context(initializer):
    read = "value[0]" if initializer.startswith("[") else "value"
    module = compile_source(f'''\
import threading
def root():
    value = {initializer}
    def worker():
        return {read} + 1
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return 1
''')
    table = module.thread_table
    assert len(table.contexts) == 1
    cell, context = next(iter(table.contexts.items()))
    actual, formal, actual_members, formal_members, status = context
    assert status == "live"
    if initializer.startswith("["):
        assert actual in module.sequence_tables[cell.row[2]].sequences
        assert formal in module.sequence_tables[cell.row[3]].sequences
    assert "0 finding(s)" in concordance_report(module)


def test_worker_table_does_not_select_an_ordinary_callsite_specialization():
    module = compile_source('''\
import threading
def worker(value: int):
    return value + 1
def root():
    result = worker(1)
    thread = threading.Thread(target=worker, args=(9,))
    thread.start()
    thread.join()
    return result
''')
    table = module.thread_table
    body, = table.bodies
    worker, = table.worker_functions(body)
    assert not worker.metadata.get("source_callsite_variant", False)
    assert any(f.metadata.get("source_callsite_variant", False)
               and f.metadata.get("source_function_reference") == table.bodies[body][0]
               for f in module.functions.values())
    assert "0 finding(s)" in concordance_report(module)


def test_lambda_thread_target_uses_the_generic_closure_capture():
    from src.compiler.concordance_declarations import (
        SOURCE_THREAD_ACTIVATION, SOURCE_THREAD_BODY,
    )

    module = compile_source('''\
import threading
def root():
    value = 9
    thread = threading.Thread(target=lambda: value + 1)
    thread.start()
    thread.join()
    return 1
''')
    book = module.metadata["identity_book"]
    captures = tuple(
        book.latest_ref(SOURCE_CLOSURE_CAPTURE, row)
        for row in book.page(SOURCE_CLOSURE_CAPTURE).rows()
    )
    assert len(captures) == 1
    bodies = tuple(
        book.page(SOURCE_THREAD_BODY).latest(row)
        for row in book.page(SOURCE_THREAD_BODY).rows()
    )
    assert len(bodies) == 1
    assert bodies[0][2] == captures
    assert len(tuple(book.page(SOURCE_THREAD_ACTIVATION).rows())) == 1
    assert "0 finding(s)" in concordance_report(module)
