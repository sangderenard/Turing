"""Probe source ingestion of two ordinary threads using one shared cell."""

import pytest

from src.compiler.concordance_declarations import SOURCE_DISPATCH_TARGET
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


SOURCE = """\
import threading

def root():
    shared = [0]
    def worker():
        shared[0] = shared[0] + 1
    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    second.start()
    first.join()
    second.join()
    return shared[0]
"""

CONTRACT = ExtractionContract("extraction_contracts/program_extraction.yaml")
OPERATIONS = (
    "thread_create", "thread_create",
    "thread_start", "thread_start",
    "thread_join", "thread_join",
)


def test_source_graph_retains_both_thread_lifecycles():
    captured = []
    lower_ast_source_to_ssa(
        SOURCE, "root", extraction_contract=CONTRACT,
        stop_after_compilation_unit_plan=True,
        resolved_process_graph_sink=captured.append,
        progress_bars=False,
    )
    graphs = {
        entry.graph.G.graph.get("function_name"): entry.graph
        for entry in captured[0].function_table
        if getattr(entry, "graph", None) is not None
    }
    assert "worker" in graphs
    root = graphs["root"]
    observed = tuple(
        data.get("attributes", {}).get("dispatch_operation", {}).get("operation")
        for _, data in root.G.nodes(data=True)
        if data.get("attributes", {}).get("dispatch_operation")
    )
    assert observed == OPERATIONS
    worker_ref = captured[0].function_table.entry("worker").reference.address
    targets = [
        data for _, data in root.G.nodes(data=True)
        if data.get("type") == "StaticReference"
        and (data.get("attributes") or {}).get("first_class_function_ref") == worker_ref
    ]
    assert len(targets) == 1
    worker_inputs = [
        data.get("attributes") or {}
        for _, data in graphs["worker"].G.nodes(data=True)
        if data.get("type") == "Input"
    ]
    assert any(
        attributes.get("binding_name") == "shared"
        and attributes.get("binding_kind") == "closure"
        for attributes in worker_inputs
    )


def test_ssa_thread_target_identity_reads_the_concordance():
    module, _, _ = lower_ast_source_to_ssa(
        SOURCE, "root", extraction_contract=CONTRACT, progress_bars=False,
    )
    creates = [
        instruction
        for block in module.functions["root__root"].blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("dispatch_operation") == "thread_create"
    ]
    assert len(creates) == 2
    book = module.metadata["identity_book"]
    page = book.page(SOURCE_DISPATCH_TARGET)
    facts = tuple(page.latest(row) for row in page.rows())
    assert len(facts) == 2
    assert facts[0] == facts[1]
    assert all(
        instruction.attributes["target_function_ref"] == facts[0]
        for instruction in creates
    )


@pytest.mark.xfail(
    strict=True,
    reason="thread lifecycle Dispatch survives SSA, but the worker's shared-cell access does not",
)
def test_ssa_retains_both_threads_and_the_shared_cell_access():
    module, _, _ = lower_ast_source_to_ssa(
        SOURCE, "root", extraction_contract=CONTRACT, progress_bars=False,
    )
    root = module.functions["root__root"]
    operations = tuple(
        instruction.attributes["dispatch_operation"]
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Dispatch"
    )
    assert operations == OPERATIONS
    assert any("worker" in name for name in module.functions), "worker function is absent from SSA"
    assert any(
        instruction.op in {"GetElementPtr", "Load", "Store", "Indexed"}
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    ), "no SSA operation accesses the shared cell"
    returned = [
        instruction.args[0].id
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret" and instruction.args
    ]
    assert returned
    assert not any(
        instruction.op == "Const" and instruction.res is not None
        and instruction.res.id in returned
        for block in root.blocks.values()
        for instruction in block.instrs
    ), "root folded the post-join shared read to a constant"
