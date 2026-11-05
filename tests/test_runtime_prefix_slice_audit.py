"""The authored prefix and its clipped borrowed view have one booked identity."""

from pathlib import Path
from dataclasses import replace

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.concordance_declarations import (
    PROCESS_GRAPH_OPERANDS, SOURCE_TENSOR_PREFIX_SLICE,
    TENSOR_METADATA_RESULT_BINDING,
    TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE, identity_book
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


@pytest.fixture(scope="module")
def prefix():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "membership", "storage": "span",
             "dtype": "int64", "rank": 2,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "selected_count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
        ], "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        "def root(membership: AbstractTensor, selected_count: int):\n"
        "    selected = membership[:selected_count]\n"
        "    return selected.sum(dim=0).sum().item()\n",
        "root", name="runtime_prefix_slice_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    source_row, = book.page(SOURCE_TENSOR_PREFIX_SLICE).rows()
    runtime_row, = book.page(TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY).rows()
    assert CorrelationTable._source_tensor_prefix_slice_findings(module) == []
    assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module) == []
    assert CorrelationTable._tensor_metadata_result_binding_findings(module) == []
    return module, book, source_row, runtime_row


def test_prefix_source_requires_authored_indexed_operand(prefix):
    module, book, row, _ = prefix
    operand = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
    original = book.page(PROCESS_GRAPH_OPERANDS).cells[(operand.row, operand.column)]
    book.page(PROCESS_GRAPH_OPERANDS).cells[(operand.row, operand.column)] = ()
    try:
        assert CorrelationTable._source_tensor_prefix_slice_findings(module)
    finally:
        book.page(PROCESS_GRAPH_OPERANDS).cells[(operand.row, operand.column)] = original


def test_prefix_source_rejects_foreign_upper(prefix):
    module, book, row, _ = prefix
    receipt = book.latest_ref(SOURCE_TENSOR_PREFIX_SLICE, row)
    page = book.page(SOURCE_TENSOR_PREFIX_SLICE)
    original = page.cell(receipt.row, receipt.column)
    page.cells[(receipt.row, receipt.column)] = (original[0], original[1], -1)
    try:
        assert CorrelationTable._source_tensor_prefix_slice_findings(module)
    finally:
        page.cells[(receipt.row, receipt.column)] = original


@pytest.mark.parametrize("opcode", ("Lt", "Add", "Select", "Max", "Min"))
def test_prefix_runtime_requires_each_clipping_step(prefix, opcode):
    module, _, _, row = prefix
    function = module.functions[row[1]]
    candidates = [i for b in function.blocks.values() for i in b.instrs
                  if i.op == opcode and i.res is not None]
    if opcode == "Select":
        consumed_by_max = {a.id for b in function.blocks.values()
                           for i in b.instrs if i.op == "Max" for a in i.args}
        candidates = [i for i in candidates if i.res.id in consumed_by_max]
    instruction, = candidates
    old = instruction.op
    instruction.op = "Sub"
    try:
        assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)
    finally:
        instruction.op = old


def test_prefix_runtime_requires_trailing_shape_store(prefix):
    module, _, _, row = prefix
    function = module.functions[row[1]]
    stores = [i for b in function.blocks.values() for i in b.instrs
              if i.op == "Store" and i.attributes.get("binding") == "prefix-slice-shape"]
    assert len(stores) == 2
    old = stores[-1].op
    stores[-1].op = "Nop"
    try:
        assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)
    finally:
        stores[-1].op = old


def test_prefix_runtime_rejects_separate_storage(prefix):
    module, book, _, row = prefix
    table = module.tensor_tables[row[1]]
    descriptor = table.by_id(row[2])
    table.tensors[row[2]] = replace(descriptor, alias_of=-1)
    try:
        assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)
    finally:
        table.tensors[row[2]] = descriptor


def test_prefix_runtime_requires_source_page_cause(prefix, monkeypatch):
    module, book, source_row, runtime_row = prefix
    receipt = book.latest_ref(TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY, runtime_row)
    source = book.latest_ref(SOURCE_TENSOR_PREFIX_SLICE, source_row)
    assert source in {s for s, _ in book.edges_into(receipt)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == receipt.key and edge[1] == source.key)
    ))
    assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)


def test_prefix_runtime_requires_safe_offset_cause(prefix, monkeypatch):
    module, book, _, row = prefix
    function = module.functions[row[1]]
    add, = (i for b in function.blocks.values() for i in b.instrs
            if i.op == "Add" and i.res is not None)
    offset = ssa_value_identity_cell(function, add.args[0].id, book=book)
    receipt = book.latest_ref(TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY, row)
    assert offset in {source for source, _ in book.edges_into(receipt)}
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == receipt.key and edge[1] == offset.key)
    ))
    assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)


@pytest.mark.parametrize("position", (0, 1, 2, 3))
def test_prefix_metadata_requires_each_call_output_port(prefix, position):
    module, book, _, row = prefix
    result_row, = (r for r in book.page(TENSOR_METADATA_RESULT_BINDING).rows()
                   if len(r) == 4 and r[2] == row[1] and r[3] == row[2])
    caller = module.functions[next(name for name, f in module.functions.items()
                                   if f.name.startswith("runtime_prefix_slice_audit__root")
                                   and "planned_region" not in f.name)]
    call, = (i for b in caller.blocks.values() for i in b.instrs
             if i.op == "Call" and i.res is not None and i.res.id == result_row[1])
    old = call.attributes["callee_output_ids"]
    forged = list(old)
    forged[position] = -1
    call.attributes["callee_output_ids"] = tuple(forged)
    try:
        assert CorrelationTable._tensor_metadata_result_binding_findings(module)
    finally:
        call.attributes["callee_output_ids"] = old


def test_prefix_runtime_rejects_wrong_count_chain(prefix):
    module, _, _, row = prefix
    function = module.functions[row[1]]
    count = next(i for b in function.blocks.values() for i in b.instrs
                 if i.op == "Cast" and i.res is not None
                 and i.res.id == identity_book(module).page(
                     TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY).latest(row)[10])
    old = count.args[0]
    count.args[0] = next(argument for argument in function.args
                         if argument.id == identity_book(module).page(
                             TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY).latest(row)[0])
    try:
        assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)
    finally:
        count.args[0] = old


@pytest.mark.parametrize("opcode", ("LOr", "Cast"))
def test_prefix_runtime_requires_zero_seed_and_checked_count(prefix, opcode):
    module, book, _, row = prefix
    function = module.functions[row[1]]
    fact = book.page(TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY).latest(row)
    if opcode == "LOr":
        instruction, = (i for b in function.blocks.values() for i in b.instrs
                        if i.op == "LOr")
        original = instruction.op
        instruction.op = "Or"
    else:
        instruction, = (i for b in function.blocks.values() for i in b.instrs
                        if i.op == "Cast" and i.res is not None
                        and i.res.id == fact[10])
        original = dict(instruction.attributes)
        instruction.attributes["overflow"] = "wrap"
    try:
        assert CorrelationTable._tensor_runtime_prefix_slice_geometry_findings(module)
    finally:
        if opcode == "LOr":
            instruction.op = original
        else:
            instruction.attributes.clear()
            instruction.attributes.update(original)
