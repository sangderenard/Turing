"""A runtime axis view retains its exact geometry and source evidence."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    SSA_VALUE, TENSOR_METADATA_RESULT_BINDING, TENSOR_RUNTIME_VIEW_GEOMETRY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE, identity_book


def _capture_view(source, rank):
    # Capture the real source writer at publication. The full-source tests
    # separately check later frame/type transformations; this is not a
    # completed source/native product or a bypass of its final audit.
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits", "storage": "span",
                    "dtype": "int64", "rank": rank,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    publish = tensor_ssa_lowering.publish_computed_region_result_metadata
    captured = []

    class ObservedView(Exception):
        pass

    def capture(module, **options):
        changed = publish(module, **options)
        if tuple(identity_book(module).page(TENSOR_RUNTIME_VIEW_GEOMETRY).rows()):
            captured.append(module)
            raise ObservedView
        return changed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tensor_ssa_lowering, "publish_computed_region_result_metadata", capture)
        with pytest.raises(ObservedView):
            lower_ast_source_to_ssa(
                source, "root", name="runtime_view_audit",
                extraction_contract=contract,
                python_bindings={"AbstractTensor": AbstractTensor},
                tensor_ssa_reference=c_backend_repository_ssa_reference(),
                progress=lambda _: None, progress_bars=False,
            )
    module, = captured
    assert not CorrelationTable._tensor_runtime_view_geometry_findings(module)
    return module


@pytest.fixture(scope="module")
def view_module():
    return _capture_view("""def root(bits: AbstractTensor) -> AbstractTensor:
    column = bits.unsqueeze(1)
    return column + 1
""", 1)


@pytest.fixture(scope="module")
def flatten_module():
    return _capture_view("""def root(bits: AbstractTensor) -> AbstractTensor:
    flat = bits.flatten()
    return flat + 1
""", 2)


def test_runtime_flatten_audit_requires_exact_source_count(flatten_module):
    module = flatten_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_VIEW_GEOMETRY)
    row, = tuple(page.rows())
    fact = page.latest(row)
    assert fact[0] == "flatten"
    function = module.functions[row[1]]
    definitions = {int(instruction.res.id): instruction
                   for block in function.blocks.values() for instruction in block.instrs
                   if instruction.res is not None}
    store, = tuple(instruction for block in function.blocks.values()
                   for instruction in block.instrs if instruction.op == "Store"
                   and definitions[int(instruction.args[1].id)].op == "GetElementPtr"
                   and int(definitions[int(instruction.args[1].id)].args[0].id) == fact[5])
    original = store.args[0]
    store.args[0] = definitions[fact[3]].res  # rank is not element count
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        store.args[0] = original


def test_runtime_view_audit_rejects_wrong_axis_and_physical_rank(view_module):
    module = view_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_VIEW_GEOMETRY)
    row, = tuple(page.rows())
    ref = book.latest_ref(TENSOR_RUNTIME_VIEW_GEOMETRY, row)
    fact = page.cell(ref.row, ref.column)
    page.cells[(ref.row, ref.column)] = (*fact[:3], 0, *fact[4:])
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = fact

    source = book.latest_ref(SSA_VALUE, (row[0], row[2]))
    values = book.page(SSA_VALUE)
    original = values.cell(source.row, source.column)
    values.cells[(source.row, source.column)] = replace(original, rank=1)
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        values.cells[(source.row, source.column)] = original


def test_runtime_view_audit_rejects_nonunit_inserted_axis(view_module):
    module = view_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_VIEW_GEOMETRY)
    row, = tuple(page.rows())
    fact = page.latest(row)
    function = module.functions[row[1]]
    definitions = {int(instruction.res.id): instruction
                   for block in function.blocks.values() for instruction in block.instrs
                   if instruction.res is not None}
    stores = {}
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.op != "Store":
                continue
            address = definitions[int(instruction.args[1].id)]
            if address.op == "GetElementPtr" and int(address.args[0].id) == fact[7]:
                index = definitions[int(address.args[1].id)].attributes["constant"]
                stores[index] = instruction
    inserted = stores[fact[3]]
    copied = next(store for index, store in stores.items() if index != fact[3])
    original = inserted.args[0]
    inserted.args[0] = copied.args[0]
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        inserted.args[0] = original

    value_cell = book.latest_ref(SSA_VALUE, (row[0], int(copied.args[0].id)))
    values = book.page(SSA_VALUE)
    old_fact = values.cell(value_cell.row, value_cell.column)
    values.cells[(value_cell.row, value_cell.column)] = replace(
        old_fact, physical_dtype="int64",
    )
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        values.cells[(value_cell.row, value_cell.column)] = old_fact


def test_runtime_view_metadata_result_requires_its_axis_proof(view_module):
    module = view_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_VIEW_GEOMETRY)
    row, = tuple(page.rows())
    proof = book.latest_ref(TENSOR_RUNTIME_VIEW_GEOMETRY, row)
    result_page = book.page(TENSOR_METADATA_RESULT_BINDING)
    binding, = tuple(binding for binding in result_page.rows()
                     if binding[2:] == (row[1], row[2]))
    result = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, binding)
    edge = next((result.key, cause.key, stage.name)
                for cause, stage in book.edges_into(result) if cause == proof)
    edges = book.page(EDGE_PAGE)
    fact = edges.cells.pop((edge, 0))
    stamp = edges.stamps.pop((edge, 0))
    columns = edges.row_columns.pop(edge)
    edges.scopes[result.key].pop(edge)
    try:
        assert CorrelationTable._tensor_runtime_view_geometry_findings(module)
    finally:
        edges.cells[(edge, 0)] = fact
        edges.stamps[(edge, 0)] = stamp
        edges.row_columns[edge] = columns
        edges.scopes[result.key][edge] = None
