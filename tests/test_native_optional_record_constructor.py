"""Record constructors preserve the distinction between absence and zero."""

from dataclasses import dataclass

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_self_check import run_all


@dataclass
class OptionalResult:
    amount: float | None = None


@pytest.mark.parametrize("argument", ["", "None", "value"])
def test_native_optional_constructor_returns_presence_and_payload(tmp_path, argument):
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {"OptionalResult": {
            "identity": f"{__name__}.OptionalResult",
            "fields": {"amount": {
                "storage": "scalar", "dtype": "float64", "optional": True,
                "default": None,
            }},
        }},
        "bindings": [],
        "values": [{"function": "root", "parameter": "value", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "python_type": "builtins.float"}],
    })
    module, _, _ = lower_ast_source_to_ssa(
        f"def root(value):\n    return OptionalResult({argument})\n",
        "root", name="optional_record", python_bindings={"OptionalResult": OptionalResult},
        extraction_contract=contract,
    )
    assert run_all(module) == []
    root = module.functions["optional_record__root"]
    record = next(iter(module.record_tables[root.name].records.values()))
    fields = {field.name: field for field in record.fields}
    presence_id = fields["amount.__present"].value_ids[0]
    payload_id = fields["amount"].value_ids[0]
    artifact = emit_ssa_to_c(module, root.name, entry_name="optional_record_native")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    for value in (0.0, 2.5):
        feeds = {int(formal.id): value for formal in root.args}
        execution = artifact.prepare_execution(feeds).run()
        assert bool(execution.buffers[presence_id].item()) == (argument == "value")
        assert execution.buffers[payload_id].item() == (value if argument == "value" else 0.0)


def test_native_record_none_test_reads_local_presence_descriptor(tmp_path):
    from src.compiler.ssa_optional_values import lower_optional_scalar_returns
    from src.transmogrifier.ssa import (
        BasicBlock, Function, Instr, IRModule, SSAValue,
        SSARecordTable, SSARecordDescriptor, SSARecordFieldDescriptor,
    )

    payload = SSAValue(0, "float64", accounting={"ssa_optional_presence_id": 999})
    present, absent, compared = SSAValue(1, "bool"), SSAValue(2, "none"), SSAValue(3, "bool")
    function = Function("root", [payload, present], {"entry": BasicBlock("entry", [
        Instr("NoneValue", [], absent), Instr("Eq", [payload, absent], compared),
        Instr("Ret", [compared], None),
    ])}, metadata={"parameter_names": (("payload", 0), ("present", 1))})
    records = SSARecordTable()
    records.register(SSARecordDescriptor(9, "OptionalResult", (
        SSARecordFieldDescriptor("amount", "scalar", value_ids=(0,), dtype="float64"),
        SSARecordFieldDescriptor("amount.__present", "scalar", value_ids=(1,), dtype="bool"),
    )))
    module = IRModule({"root": function}, record_tables={"root": records})
    lower_optional_scalar_returns(module)
    assert function.blocks["entry"].instrs[1].args == [present]
    artifact = emit_ssa_to_c(module, "root", entry_name="optional_field_test")
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path, optimization="O0")
    for value, presence in ((0.0, False), (0.0, True), (2.5, True)):
        execution = artifact.prepare_execution({0: value, 1: presence}).run()
        assert bool(execution.buffers[3].item()) is (not presence)
