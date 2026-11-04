"""Expanded cat members retain distinct stable and loop-carried bindings."""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.concordance_declarations import REGION_CAPTURE_BINDING
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable


def test_cat_keeps_stable_seed_beside_loop_carried_alias():
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": "root", "parameter": name, "storage": "span",
                    "dtype": "int64", "rank": 1, "shape": [4],
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}
                   for name in ("data", "indices")],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(data: AbstractTensor, indices: AbstractTensor, count: int) -> int:\n"
        "    leaves = data\n"
        "    current = leaves\n"
        "    for _ in range(count):\n"
        "        package = current + 1\n"
        "        combined = AbstractTensor.cat((leaves, package), dim=0)\n"
        "        current = combined[indices]\n"
        "    return current.sum().item()\n",
        "root", name="loop_cat_stable_seed",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    assert CorrelationTable.build(module).findings(module) == []
    book = module.metadata["identity_book"]
    captures = book.page("region_capture_binding")
    assert any(captures.latest(row)[1] == "leaves" for row in captures.rows())
    function = module.functions[exports[0]]
    parameters = dict(function.metadata["parameter_names"])
    cat_region, = [candidate for candidate in module.functions.values()
                   if any(instruction.op == "cat" for block in candidate.blocks.values()
                          for instruction in block.instrs)]
    calls = [instruction for block in function.blocks.values() for instruction in block.instrs
             if instruction.op == "Call"]
    cat_call, = [instruction for instruction in calls
                 if instruction.attributes.get("callee") == cat_region.name]
    assert cat_call.args[0].id == parameters["data"]
    phi_ids = {instruction.res.id for block in function.blocks.values()
               for instruction in block.instrs if instruction.op == "Phi"}
    assert cat_call.args[0].id not in phi_ids
    assert any(instruction.args and instruction.args[0].id in phi_ids
               for instruction in calls
               if instruction.attributes.get("callee") != cat_region.name)
    row = next(row for row in captures.rows() if captures.latest(row)[1] == "leaves")
    cell = book.latest_ref(REGION_CAPTURE_BINDING, row)
    key = (cell.row, cell.column)
    original = captures.cells[key]
    try:
        captures.cells[key] = (original[0], "current")
        assert CorrelationTable._region_aggregate_capture_binding_findings(module)
    finally:
        captures.cells[key] = original
    assert not CorrelationTable._region_aggregate_capture_binding_findings(module)
