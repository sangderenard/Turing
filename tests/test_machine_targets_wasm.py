import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler import machine_targets
from src.compiler.fused_program_wasm_backend import (
    WasmEmissionError,
    emit_wasm_module,
)


def _program(*steps: OpStep, feeds, outputs) -> FusedProgram:
    return FusedProgram(
        version=1, feeds=set(feeds), steps=list(steps), outputs=dict(outputs)
    )


def _sub_abs_plus_one() -> FusedProgram:
    left, right, s0, s1, s2 = 1, 2, 3, 4, 5
    return _program(
        OpStep(step_id=0, op_name="sub", input_ids=[left, right], attrs={}, result_id=s0),
        OpStep(step_id=1, op_name="abs", input_ids=[s0], attrs={}, result_id=s1),
        OpStep(
            step_id=2, op_name="add", input_ids=[s1],
            attrs={"right_scalar": 1.0}, result_id=s2,
        ),
        feeds=(left, right),
        outputs={"result": s2},
    )


def test_elementwise_program_emits_a_complete_module():
    module = emit_wasm_module(_sub_abs_plus_one(), name="demo")

    assert module.complete, module.shortfall_report()
    assert module.parameters == ("$count", "$feed0", "$feed1", "$out0")
    assert module.value_type == "f64"
    # The instructions the steps lower to, in order.
    for instruction in ("f64.sub", "f64.abs", "f64.add", "f64.load", "f64.store"):
        assert instruction in module.source, instruction
    # Structured control flow: the only loop is the elementwise walk.
    assert module.source.count("loop $body") == 1
    assert "br_if $done" in module.source


def test_transcendentals_are_reported_not_approximated():
    """WebAssembly has no exp/log/sin instruction. Emitting a polynomial
    approximation would return a plausible wrong number, so the step is
    named as a shortfall instead."""

    for operation in ("exp", "log", "sin", "pow"):
        module = emit_wasm_module(
            _program(
                OpStep(step_id=0, op_name=operation, input_ids=[1], attrs={}, result_id=2),
                feeds=(1,),
                outputs={"result": 2},
            ),
            name="t",
        )
        assert not module.complete
        assert operation in module.shortfall_report()


def test_a_scalar_operand_becomes_a_constant_not_a_second_feed():
    module = emit_wasm_module(
        _program(
            OpStep(
                step_id=0, op_name="mul", input_ids=[1],
                attrs={"right_scalar": 2.5}, result_id=2,
            ),
            feeds=(1,),
            outputs={"result": 2},
        ),
        name="t",
    )
    assert module.complete
    assert "f64.const 2.5" in module.source
    assert module.parameters == ("$count", "$feed0", "$out0")


def test_reversed_scalar_operands_keep_their_order():
    """value - tensor is not tensor - value, and the stack order is what
    decides it."""

    module = emit_wasm_module(
        _program(
            OpStep(
                step_id=0, op_name="sub", input_ids=[1],
                attrs={"right_scalar": 10.0, "reverse": True}, result_id=2,
            ),
            feeds=(1,),
            outputs={"result": 2},
        ),
        name="t",
    )
    body = module.source
    assert body.index("f64.const 10.0") < body.index("local.get $v0")


def test_comparisons_come_back_in_the_value_type():
    """Every other backend reports a comparison as 0.0/1.0 in the operand's
    type; WebAssembly's comparisons yield i32, so the result is converted."""

    module = emit_wasm_module(
        _program(
            OpStep(step_id=0, op_name="less", input_ids=[1, 2], attrs={}, result_id=3),
            feeds=(1, 2),
            outputs={"result": 3},
        ),
        name="t",
    )
    assert module.complete
    assert "f64.lt" in module.source
    assert "f64.convert_i32_u" in module.source


def test_an_unrepresentable_dtype_is_refused():
    program = _sub_abs_plus_one()
    with pytest.raises(WasmEmissionError):
        emit_wasm_module(program, name="t", dtype="int64")


def test_float32_programs_use_four_byte_strides():
    module = emit_wasm_module(_sub_abs_plus_one(), name="t", dtype="float32")
    assert module.value_type == "f32"
    assert "i32.const 4" in module.source
    assert "f32.load" in module.source


# --- the hub ---------------------------------------------------------------


def test_wasm_is_registered_and_declares_what_it_cannot_do():
    names = {c.name for c in machine_targets.capabilities()}
    assert "wasm" in names

    (wasm,) = [c for c in machine_targets.capabilities() if c.name == "wasm"]
    assert wasm.consumes == "fused_program"
    assert wasm.emits == ".wat"
    # It writes the elementwise walk itself but does not lower a program's
    # own control flow.
    assert wasm.control_flow is False
    assert {"exp", "log", "sin", "pow"} <= wasm.unsupported_operations


def test_the_hub_chooses_before_emitting_rather_than_by_failing():
    representable = _program(
        OpStep(step_id=0, op_name="mul", input_ids=[1, 2], attrs={}, result_id=3),
        feeds=(1, 2),
        outputs={"result": 3},
    )
    transcendental = _program(
        OpStep(step_id=0, op_name="sin", input_ids=[1], attrs={}, result_id=2),
        feeds=(1,),
        outputs={"result": 2},
    )

    assert "wasm" in machine_targets.targets_for(representable)
    assert "wasm" not in machine_targets.targets_for(transcendental)


def test_every_target_returns_the_same_artifact_shape():
    artifact = machine_targets.emit(
        _sub_abs_plus_one(), "wasm", name="shared_shape"
    )
    assert artifact.target == "wasm"
    assert artifact.extension == ".wat"
    assert artifact.complete
    assert artifact.api is not None
    assert artifact.source.startswith("(module")


def test_the_artifact_carries_a_calling_contract():
    """The descriptor is the point: a caller binds argument order and
    meaning from a record instead of reading emitted source."""

    artifact = machine_targets.emit(_sub_abs_plus_one(), "wasm", name="contract")
    mapping = artifact.api.to_mapping()

    assert mapping["language"] == "wasm"
    assert mapping["entry"] == "run"
    (entry,) = mapping["entry_points"]
    roles = [p["role"] for p in entry["parameters"]]
    assert roles == ["extent", "input", "input", "output"]
    assert mapping["metadata"]["element_bytes"] == 8
    assert mapping["metadata"]["memory_export"] == "memory"


def test_an_unknown_target_is_named_not_guessed():
    with pytest.raises(KeyError, match="definitely_not_a_target"):
        machine_targets.get_target("definitely_not_a_target")


def test_assembly_availability_is_separate_from_emission():
    """Emission never needs a toolchain; assembly does. A missing assembler
    must not make the target look unusable."""

    target = machine_targets.get_target("wasm")
    artifact = target.emit(_sub_abs_plus_one(), name="regardless")
    assert artifact.complete  # emitted fine
    assert isinstance(target.available(), bool)  # independent question


def test_writing_an_artifact_puts_the_descriptor_beside_it(tmp_path):
    artifact = machine_targets.emit(_sub_abs_plus_one(), "wasm", name="beside")
    path = artifact.write(tmp_path)

    assert path.name == "beside.wat"
    assert path.with_suffix(".api.yaml").is_file()
