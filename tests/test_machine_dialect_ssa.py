from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.machine_dialect_ssa import (
    MACHINE_SSA_DIALECT, decoded_function_to_machine_ssa,
    machine_dialect_occurrences, repository_ssa_legalized,
)
from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder
from src.compiler.ssa_fortran_backend import (
    FortranEmissionError,
    emit_function as emit_fortran_function,
    emit_module,
)
from src.compiler.ssa_spirv_backend import (
    SPIRVEmissionError, emit_function as emit_spirv_function,
)
from src.compiler.ssa_webgpu_backend import (
    WGSLEmissionError, emit_module as emit_wgsl_module,
)
from src.compiler.ssa_webgl_backend import (
    SSAWebGLEmissionError,
    emit_ssa_webgl_fragment_module,
)
from src.transmogrifier.ssa import IRModule
import pytest


def _decode_all(encoded: bytes):
    return X86ReferenceDecoder().decode_report(
        encoded, base_address=0x1000,
        stop_at_return=False, allow_trailing_after_terminal=True,
    ).instructions


def test_machine_dialect_retains_loop_with_state_phi():
    # test eax,eax; jne back to test; ret
    function = decoded_function_to_machine_ssa(
        "loop", _decode_all(b"\x85\xc0\x75\xfc\xc3"),
    )

    assert function.metadata["dialect"] == MACHINE_SSA_DIALECT
    preheader = function.blocks["entry"]
    header = function.blocks["block_0000000000001000"]
    latch = function.blocks["block_0000000000001004"]
    assert preheader.successors == ["block_0000000000001000"]
    assert header.instrs[0].op == "machine.PhiState"
    assert header.instrs[0].attributes["incoming_blocks"] == (
        "entry", "block_0000000000001000",
    )
    assert any(item.op == "machine.integer_test" for item in header.instrs)
    assert any(item.op == "CondBr" for item in header.instrs)
    assert header.successors == [
        "block_0000000000001000", "block_0000000000001004",
    ]
    assert latch.instrs[0].op == "machine.PhiState"
    assert not repository_ssa_legalized(function)
    assert machine_dialect_occurrences(function)


def test_fortran_emitter_rejects_machine_dialect_container_reuse():
    function = decoded_function_to_machine_ssa(
        "loop", _decode_all(b"\x85\xc0\x75\xfc\xc3"),
    )

    with pytest.raises(FortranEmissionError) as failure:
        emit_module(IRModule({"loop": function}), extra_roots=("loop",))

    message = str(failure.value)
    assert "requires legalized repository SSA" in message
    assert "machine.PhiState" in message

    with pytest.raises(FortranEmissionError, match="machine.PhiState"):
        emit_fortran_function(function)


def test_gpu_emitters_reject_machine_dialect_container_reuse():
    function = decoded_function_to_machine_ssa(
        "loop", _decode_all(b"\x85\xc0\x75\xfc\xc3"),
    )
    module = IRModule({"loop": function})

    with pytest.raises(WGSLEmissionError, match="machine.PhiState"):
        emit_wgsl_module(module, outputs={"loop": ()})
    with pytest.raises(SPIRVEmissionError, match="machine.PhiState"):
        emit_spirv_function(function)
    with pytest.raises(SSAWebGLEmissionError, match="machine.PhiState"):
        emit_ssa_webgl_fragment_module(function)


def test_semantic_family_test_legalizes_to_ordinary_ssa():
    result = raise_binary_region_to_ssa(
        b"\x48\x85\xc0\xc3",
        maximum_file_size=4,
        size=4,
        base_address=0x1000,
        name="test_then_return",
        full_vocabulary_report=True,
    )

    assert result.function is not None
    assert "dialect" not in result.function.metadata
    assert not result.failed_vocabulary
    assert any(
        instruction.attributes.get("machine_flag") == "ZF"
        for block in result.function.blocks.values()
        for instruction in block.instrs
    )
    assert result.complete is True
    assert repository_ssa_legalized(result.function)
    assert machine_dialect_occurrences(result.function) == ()


def test_conditional_preserves_local_and_external_branch_order():
    # jne outside region; ret fallthrough remains local
    function = decoded_function_to_machine_ssa(
        "external_true", _decode_all(b"\x75\x7f\xc3"),
    )
    branch = next(
        item
        for item in function.blocks["block_0000000000001000"].instrs
        if item.op == "CondBr"
    )

    assert branch.attributes["true_target"] is None
    assert branch.attributes["true_target_address"] == 0x1081
    assert branch.attributes["false_target"] == "block_0000000000001002"
    assert branch.attributes["false_target_address"] is None
