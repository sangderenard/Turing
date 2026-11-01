"""Synchronization is compiled from authored source, never hand-made SSA."""

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.synchronization import Scope


CONTRACT = ExtractionContract("extraction_contracts/program_extraction.yaml")
SOURCE = """\
from src.compiler.synchronization import (
    Scope, MemorySemantics, memory_barrier, compiler_fence, control_barrier,
    control_memory_barrier, latch_create, latch_arrive, latch_wait, latch_reset, latch_signal,
)
def root():
    shared = [0]
    gate = latch_create(1, 0, Scope.THREAD_REGION)
    memory_barrier(Scope.THREAD_REGION, MemorySemantics.ACQUIRE, shared)
    compiler_fence(MemorySemantics.SEQUENTIALLY_CONSISTENT, shared)
    control_barrier(gate, Scope.THREAD_REGION, 0, 1)
    latch_reset(gate, 0, 1, 1)
    control_memory_barrier(gate, Scope.THREAD_REGION, Scope.THREAD_REGION,
                           MemorySemantics.ACQUIRE_RELEASE | MemorySemantics.WORKGROUP_MEMORY,
                           1, 1, shared)
    latch_reset(gate, 1, 2, 1)
    latch_arrive(gate, 2, 1)
    latch_wait(gate, 2)
    latch_reset(gate, 2, 3, 1)
    latch_signal(gate, 3)
    latch_wait(gate, 3)
    return 1
"""


def compile_source(source):
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", extraction_contract=CONTRACT,
        progress=lambda _: None, progress_bars=False,
    )
    return module


def instructions(module):
    return [i for b in module.functions["root__root"].blocks.values() for i in b.instrs]


def synchronization(module):
    return [i for i in instructions(module) if i.attributes.get("synchronization_receipt")]


def test_all_barrier_and_latch_kinds_reach_ssa_with_ordered_activity():
    module = compile_source(SOURCE)
    sync = synchronization(module)
    assert [i.op for i in sync] == [
        "LatchCreate", "MemoryBarrier", "MemoryBarrier", "ControlBarrier", "LatchReset",
        "MemoryBarrier", "ControlBarrier", "MemoryBarrier", "LatchReset", "LatchArrive",
        "LatchWait", "LatchReset", "LatchSignal", "LatchWait",
    ]
    for previous, current in zip(sync, sync[1:]):
        assert current.arg_roles[-1] == "after"
        assert current.args[-1] is previous.res
    assert all(i.attributes["ordered_effect"] and i.attributes["effectful"] for i in sync)
    storage = {i.args[i.arg_roles.index("storage")].id for i in sync if "storage" in i.arg_roles}
    assert len(storage) == 1
    assert CorrelationTable._synchronization_findings(module) == []


@pytest.mark.parametrize("semantics, phases, orders", [
    (2, ["rendezvous", "acquire"], [2]),
    (4, ["release", "rendezvous"], [4]),
    (8, ["release", "rendezvous", "acquire"], [4, 2]),
    (16, ["release", "rendezvous", "acquire"], [16, 16]),
])
def test_combined_barrier_preserves_one_identity_and_both_sides_of_rendezvous(semantics, phases, orders):
    source = f"""\
from src.compiler.synchronization import latch_create, control_memory_barrier
def root():
    shared = [0]
    gate = latch_create(1, 0, 256)
    control_memory_barrier(gate, 256, 256, {semantics}, 0, 1, shared)
    return 1
"""
    module = compile_source(source)
    sync = [i for i in synchronization(module) if i.attributes["synchronization_kind"] == "control_memory_barrier"]
    assert [i.attributes["synchronization_phase"] for i in sync] == phases
    assert len({i.attributes["synchronization_contract"] for i in sync}) == 1
    literals = {i.res.id: i.attributes["value"] for i in instructions(module) if i.op == "Const" and i.res is not None}
    assert [literals[i.args[i.arg_roles.index("memory_semantics")].id] for i in sync if i.op == "MemoryBarrier"] == orders


def test_memory_fences_keep_every_explicit_scope_and_order_without_rendezvous():
    calls = "\n".join(f"    memory_barrier({int(scope)}, {order}, shared)"
                      for scope in Scope for order in (2, 4, 8, 16))
    module = compile_source("from src.compiler.synchronization import memory_barrier\ndef root():\n    shared = [0]\n" + calls + "\n    return 1\n")
    sync = synchronization(module)
    assert len(sync) == len(Scope) * 4
    assert all(i.op == "MemoryBarrier" and not i.attributes["requires_concurrent_progress"] for i in sync)
    literals = {i.res.id: i.attributes["value"] for i in instructions(module) if i.op == "Const" and i.res is not None}
    assert [(literals[i.args[0].id], literals[i.args[1].id]) for i in sync] == [
        (int(scope), order) for scope in Scope for order in (2, 4, 8, 16)
    ]


def test_composed_memory_mask_is_defined_before_the_fence_that_consumes_it():
    source = """\
from src.compiler.synchronization import Scope, MemorySemantics, memory_fence
def root():
    shared = [0]
    memory_fence(memory_semantics=MemorySemantics.ACQUIRE_RELEASE | MemorySemantics.WORKGROUP_MEMORY,
                 storage=shared, memory_scope=Scope.THREAD_REGION)
    return 1
"""
    module = compile_source(source)
    (fence,) = synchronization(module)
    assert fence.op == "MemoryBarrier"
    stream = instructions(module)
    definitions = {i.res.id: index for index, i in enumerate(stream) if i.res is not None}
    definitions.update((value.id, -1) for value in module.functions["root__root"].args)
    assert all(definitions[value.id] < stream.index(fence) for value in fence.args)
    from src.compiler.concordance_declarations import SOURCE_SYNCHRONIZATION
    contract = fence.attributes["synchronization_contract"]
    fact = module.metadata["identity_book"].page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
    assert dict(fact[3])["memory_semantics"] == 264


CONTROL_OWNERSHIP_SOURCE = """\
from src.compiler.synchronization import memory_barrier, latch_create, control_barrier, latch_reset
def root(flag: bool, count: int):
    shared = [0]
    if flag:
        memory_barrier(256, 4, shared)
    else:
        memory_barrier(256, 2, shared)
    gate = latch_create(1, 0, 256)
    for generation in range(count):
        control_barrier(gate, 256, generation, 1)
        latch_reset(gate, generation, generation + 1, 1)
    return 1
"""


def test_synchronization_keeps_conditional_and_repeated_control_ownership():
    source = CONTROL_OWNERSHIP_SOURCE
    module = compile_source(source)
    function = module.functions["root__root"]
    locations = [(block.name, i) for block in function.blocks.values() for i in block.instrs
                 if i.attributes.get("synchronization_receipt")]
    memory_blocks = [name for name, i in locations if i.op == "MemoryBarrier"]
    assert len(memory_blocks) == 2 and memory_blocks[0] != memory_blocks[1]
    assert any(i.op == "ControlBarrier" for _name, i in locations)
    assert CorrelationTable._synchronization_findings(module) == []


@pytest.mark.parametrize("scope, semantics, error", [
    (999, 8, "unknown memory_scope"), (256, 0, "one valid"),
    (256, 6, "one valid"), (256, 0x20, "one valid"),
    (256, 0x2002, "MakeAvailable"), (256, 0x4004, "MakeVisible"),
])
def test_invalid_fences_refuse_instead_of_broadening_scope(scope, semantics, error):
    source = f"from src.compiler.synchronization import memory_barrier\ndef root():\n    shared = [0]\n    memory_barrier({scope}, {semantics}, shared)\n    return 1\n"
    with pytest.raises(ValueError, match=error):
        compile_source(source)


def test_completion_wait_is_not_an_all_participant_barrier():
    source = """\
import threading
from src.compiler.synchronization import completion_wait
def root():
    shared = [0]
    def worker():
        return 1
    task = threading.Thread(target=worker)
    task.start()
    completion_wait(task, 256, 2, shared)
    task.join()
    return 1
"""
    module = compile_source(source)
    sync = synchronization(module)
    assert [i.op for i in sync] == ["CompletionWait", "MemoryBarrier"]
    assert "participants" not in sync[0].arg_roles
    creates = [i for i in instructions(module) if i.attributes.get("dispatch_operation") == "thread_create"]
    assert sync[0].args[0] is creates[0].res


def test_each_memory_class_and_visibility_flag_remains_an_explicit_operand():
    masks = [8 | flag for flag in (0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)]
    source = "from src.compiler.synchronization import memory_barrier\ndef root():\n    shared = [0]\n"
    source += "\n".join(f"    memory_barrier(256, {mask}, shared)" for mask in masks)
    source += "\n    return 1\n"
    module = compile_source(source)
    literals = {i.res.id: i.attributes["value"] for i in instructions(module) if i.op == "Const" and i.res is not None}
    assert [literals[i.args[1].id] for i in synchronization(module)] == masks


def test_an_unrelated_same_spelled_function_is_not_a_synchronization_intrinsic():
    module = compile_source("""\
def memory_barrier(scope, semantics, storage):
    return 17
def root():
    return memory_barrier(256, 8, 0)
""")
    assert synchronization(module) == []


def test_authored_fence_preserves_the_actual_post_join_shared_read():
    from tests.test_python_two_threads_shared_location import SOURCE as THREAD_SOURCE
    source = "from src.compiler.synchronization import memory_barrier\n" + THREAD_SOURCE.replace(
        "    return shared[0]", "    memory_barrier(256, 2, shared)\n    return shared[0]",
    )
    module = compile_source(source)
    stream = instructions(module)
    returned = {i.args[0].id for i in stream if i.op == "Ret" and i.args}
    assert not any(i.op == "Const" and i.res is not None and i.res.id in returned for i in stream)
    assert any(i.op == "Load" and i.res is not None and i.res.id in returned for i in stream)
