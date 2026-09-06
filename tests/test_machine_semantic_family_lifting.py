import pytest
import struct

from src.compiler.amd64_machine_semantics import (
    PagedByteMemory, default_effect_handlers,
)
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState,
)
from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.machine_dialect_ssa import repository_ssa_legalized
from src.compiler.machine_reference_vocabulary import X86InstructionToken
from src.compiler.x86_tensor_read_head import (
    X86EncodingFields,
    controlled_x86_64_read_head_profile,
)
from types import SimpleNamespace


def _canonical_encoding(row):
    modrm = None
    if row.has_modrm:
        extension = 0 if row.modrm_extension is None else row.modrm_extension
        mod = 0 if row.modrm_memory_only or row.token in {
            int(X86InstructionToken.LEA_R32_M),
            int(X86InstructionToken.LEA_R64_M),
        } else 3
        modrm = (mod << 6) | (extension << 3)
    return X86EncodingFields(
        modrm=modrm,
        immediate=0 if row.immediate_bytes else None,
    )


def test_every_authoritative_token_legalizes_to_repository_ssa():
    profile = controlled_x86_64_read_head_profile()
    failures = []
    for row in profile.rows:
        token_name = X86InstructionToken(row.token).name
        encoded = profile.encode(row.token, _canonical_encoding(row))
        if not row.terminal:
            encoded += b"\xc3"
        result = raise_binary_region_to_ssa(
            encoded,
            maximum_file_size=len(encoded),
            size=len(encoded),
            base_address=0x1000,
            name=f"canonical_{token_name.lower()}",
            full_vocabulary_report=True,
            cfg_decode=True,
        )
        if (
            not result.complete
            or result.function is None
            or not repository_ssa_legalized(result.function)
        ):
            failures.append((
                token_name,
                encoded.hex(" "),
                tuple(result.failed_vocabulary),
                () if result.function is None else tuple(
                    (block_name, instruction.op)
                    for block_name, block in result.function.blocks.items()
                    for instruction in block.instrs
                    if instruction.attributes.get("machine_dialect")
                    or instruction.op.startswith("machine.")
                ),
            ))

    # VEX forms are decoded and legalized by the authoritative vocabulary but
    # intentionally excluded from the legacy-prefix tensor write head until
    # that head carries explicit VEX fields.
    assert len(profile.rows) == 308
    assert len(X86InstructionToken) == 313
    assert failures == []


def test_real_vinsertf128_bytes_legalize_to_repo_ssa_lane_operations():
    encoded = bytes.fromhex("c4 e3 7d 18 c0 01 c3")
    result = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x180011822, name="vcruntime_memset_avx_lane",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert result.complete
    assert result.failed_vocabulary == ()
    assert result.function is not None
    assert repository_ssa_legalized(result.function)
    lane = next(
        item
        for block in result.function.blocks.values()
        for item in block.instrs
        if item.attributes.get("machine_vector_insert_lane")
    )
    assert lane.op == "Or"
    assert lane.attributes["lane_width"] == 128
    assert lane.attributes["lane_index"] == 1
    assert lane.attributes["vector_width"] == 256


@pytest.mark.parametrize(("encoded", "token"), (
    (bytes.fromhex("c5 fd 7f 01 c3"), "VMOVDQA_YMMM256_YMM"),
    (bytes.fromhex("aa c3"), "STOSB"),
    (bytes.fromhex("c5 fd e7 01 c3"), "VMOVNTDQ_M256_YMM"),
    (bytes.fromhex("f3 aa c3"), "REP_STOSB"),
))
def test_following_vcruntime_memset_forms_legalize(encoded, token):
    result = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x180011860, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert result.complete
    assert result.failed_vocabulary == ()
    assert result.decoded[0].token.name == token
    assert result.function is not None
    assert repository_ssa_legalized(result.function)


@pytest.mark.parametrize(("encoded", "expected"), (
    (b"\x48\x85\xc0\xc3", "TEST_RM64_R64"),
    (b"\x89\x01\xc3", "MOV_RM32_R32"),
    (b"\x33\xc0\xc3", "XOR_R32_RM32"),
    (b"\xb8\x01\x00\x00\x00\xc3", "MOV_R32_IMM32"),
    (b"\xf7\xc0\x01\x00\x00\x00\xc3", "TEST_RM32_IMM32"),
    (b"\xc7\x01\x01\x00\x00\x00\xc3", "MOV_RM32_IMM32"),
    (b"\x85\xc0\xc3", "TEST_RM32_R32"),
    (b"\x83\x39\x01\xc3", "CMP_RM32_IMM8"),
    (b"\x48\x63\x01\xc3", "MOVSXD_R64_RM32"),
    (b"\x48\x83\x39\x01\xc3", "CMP_RM64_IMM8"),
    (b"\x48\x81\xe8\x01\x00\x00\x00\xc3", "SUB_R64_IMM32"),
    (b"\x83\x29\x01\xc3", "SUB_RM32_IMM8"),
    (b"\x90\xc3", "NOP"),
    (b"\x48\xc7\x01\x01\x00\x00\x00\xc3", "MOV_RM64_IMM32"),
    (b"\x83\x09\x01\xc3", "OR_RM32_IMM8"),
    (b"\x48\x2b\xc1\xc3", "SUB_R64_RM64"),
    (b"\x48\x03\xc1\xc3", "ADD_R64_RM64"),
    (b"\x48\x81\xc0\x01\x00\x00\x00\xc3", "ADD_R64_IMM32"),
    (b"\x03\xc1\xc3", "ADD_R32_RM32"),
    (b"\x02\xc0\xc3", "ADD_R8_RM8"),
    (b"\x81\x21\xff\x00\x00\x00\xc3", "AND_RM32_IMM32"),
    (b"\x83\x21\x7f\xc3", "AND_RM32_IMM8"),
    (b"\x3c\x01\xc3", "CMP_AL_IMM8"),
    (b"\x2c\x3a\xc3", "SUB_AL_IMM8"),
    (b"\xa8\x01\xc3", "TEST_AL_IMM8"),
    (b"\x48\x83\x09\x01\xc3", "OR_RM64_IMM8"),
    (b"\x39\x01\xc3", "CMP_RM32_R32"),
))
def test_cached_semantic_families_legalize_without_machine_fallback(encoded, expected):
    result = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=f"lift_{expected.lower()}",
        full_vocabulary_report=True,
    )

    assert [item.token.name for item in result.decoded][0] == expected
    assert result.failed_vocabulary == ()
    assert result.function is not None
    assert "dialect" not in result.function.metadata


def test_scasb_legalizes_exact_flags_and_direction_without_rcx_state():
    result = raise_binary_region_to_ssa(
        b"\xae\xc3", maximum_file_size=2, size=2,
        base_address=0x1000, full_vocabulary_report=True,
    )
    assert result.complete and result.function is not None
    assert repository_ssa_legalized(result.function)
    instructions = [
        instruction
        for block in result.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_address") == 0x1000
    ]
    assert any(
        instruction.op == "Load"
        and instruction.attributes.get("width") == 8
        for instruction in instructions
    )
    assert {
        instruction.attributes.get("machine_flag")
        for instruction in instructions
    } >= {"CF", "PF", "AF", "ZF", "SF", "OF"}
    advance = next(
        instruction for instruction in instructions
        if instruction.attributes.get("machine_string_compare") == "scasb"
        and instruction.attributes.get("machine_register") == "RDI"
    )
    assert advance.op == "Add"
    assert not any(
        instruction.attributes.get("machine_register") == "RCX"
        for instruction in instructions
    )


def test_lock_add_rm8_r8_is_atomic_and_preserves_source_register():
    result = raise_binary_region_to_ssa(
        b"\xf0\x00\x08\xc3", maximum_file_size=4, size=4,
        base_address=0x1000, full_vocabulary_report=True,
    )
    assert result.complete and result.function is not None
    assert repository_ssa_legalized(result.function)
    instructions = [
        instruction
        for block in result.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_address") == 0x1000
    ]
    observed = next(
        instruction for instruction in instructions
        if instruction.op == "AtomicExchangeAddObserved"
    )
    memory = next(
        instruction for instruction in instructions
        if instruction.op == "AtomicExchangeAddMemory"
    )
    assert observed.attributes["width"] == 8
    assert observed.attributes["ordering"] == "sequentially-consistent"
    assert memory.attributes["locked"] is True
    assert memory.attributes["source_register_unchanged"] is True
    assert not any(
        instruction.attributes.get("machine_register") == "RCX"
        for instruction in instructions
    )


def test_lock_inc_rm32_is_atomic_and_preserves_carry_flag():
    result = raise_binary_region_to_ssa(
        b"\xf0\xff\x00\xc3", maximum_file_size=4, size=4,
        base_address=0x1000, full_vocabulary_report=True,
    )
    assert result.complete and result.function is not None
    assert result.decoded[0].token.name == "LOCK_INC_RM32"
    assert repository_ssa_legalized(result.function)
    instructions = [
        instruction
        for block in result.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_address") == 0x1000
    ]
    observed = next(
        item for item in instructions
        if item.op == "AtomicExchangeAddObserved"
    )
    memory = next(
        item for item in instructions
        if item.op == "AtomicExchangeAddMemory"
    )
    assert observed.attributes["width"] == 32
    assert observed.attributes["ordering"] == "sequentially-consistent"
    assert memory.attributes["locked"] is True
    assert memory.attributes["machine_atomic_operation"] == "lock-inc"
    assert not any(
        item.attributes.get("machine_flag") == "CF"
        and item.attributes.get("machine_address") == 0x1000
        for item in instructions
    )
    assert {item.attributes.get("machine_flag") for item in instructions} >= {
        "PF", "AF", "ZF", "SF", "OF",
    }


def test_ror_rm64_cl_has_masked_dynamic_count_and_zero_preservation():
    result = raise_binary_region_to_ssa(
        b"\x48\xd3\xc8\xc3", maximum_file_size=4, size=4,
        base_address=0x1000, full_vocabulary_report=True,
    )
    assert result.complete and result.function is not None
    assert result.decoded[0].token.name == "ROR_RM64_CL"
    assert repository_ssa_legalized(result.function)
    instructions = [
        instruction
        for block in result.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_address") == 0x1000
    ]
    assert any(item.attributes.get("machine_masked_rotate_count") for item in instructions)
    assert any(item.attributes.get("machine_rotate_zero_preserves") for item in instructions)
    assert any(item.attributes.get("machine_rotate_count_one") for item in instructions)
    assert {item.attributes.get("machine_flag") for item in instructions} >= {
        "CF", "OF",
    }


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x74\x00\xc3", "JE_REL8"),
    (b"\x77\x00\xc3", "JA_REL8"),
    (b"\x7e\x00\xc3", "JLE_REL8"),
    (b"\x7d\x00\xc3", "JGE_REL8"),
    (b"\xeb\x00\xc3", "JMP_REL8"),
))
def test_branch_encoding_variants_use_semantic_control_lowering(encoded, token):
    result = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=f"lift_{token.lower()}",
        full_vocabulary_report=True,
    )

    assert result.decoded[0].token.name == token
    assert result.failed_vocabulary == ()
    assert result.function is not None


def test_arithmetic_carry_uses_explicit_unsigned_comparison_vocabulary():
    result = raise_binary_region_to_ssa(
        b"\x48\x83\xc0\x01\xc3",  # add rax, 1; ret
        maximum_file_size=5, size=5,
        base_address=0x1000, name="carry",
        full_vocabulary_report=True,
    )

    carry = next(
        instruction
        for block in result.function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get("machine_flag") == "CF"
    )
    assert carry.op == "ULt"


def test_looping_machine_cfg_uses_full_state_phis_in_ordinary_ssa():
    # add eax,1; test eax,eax; jne back to add; ret
    encoded = b"\x83\xc0\x01\x85\xc0\x75\xf9\xc3"
    result = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="loop",
        full_vocabulary_report=True,
    )

    assert result.failed_vocabulary == ()
    assert result.function.metadata["machine_loop_state_phis"] is True
    assert result.function.metadata["entry_block"] == "__machine_preheader"
    header = result.function.blocks["entry"]
    phis = [item for item in header.instrs if item.op == "Phi"]
    assert len(phis) == 16 + 1 + 7
    assert all(len(item.args) == 2 for item in phis)
    assert {item.attributes.get("machine_state") for item in phis} == {
        "register", "memory", "flags",
    }


def test_rep_stosw_is_explicit_strided_memory_fill_with_direction_state():
    encoded = b"\x66\xf3\xab\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="rep_stosw",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "REP_STOSW"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert "__machine_df" in lifting.function.metadata["argument_names"]
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    fill = next(item for item in instructions if item.op == "StridedStoreFill")
    stride = next(
        item for item in instructions
        if item.attributes.get("machine_direction_stride")
    )
    displacement = next(
        item for item in instructions
        if item.attributes.get("machine_string_displacement")
    )
    assert fill.attributes["element_width"] == 16
    assert fill.attributes["iterative"] is True
    assert len(fill.args) == 5
    assert stride.op == "Select"
    assert displacement.op == "Mul"


def test_rep_movsq_is_ordered_strided_copy_with_explicit_register_updates():
    lifting = raise_binary_region_to_ssa(
        b"\xf3\x48\xa5\xc3", maximum_file_size=4, size=4,
        base_address=0x1000, name="rep_movsq",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        instruction for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    copy = next(item for item in instructions if item.op == "StridedMemoryCopy")
    assert copy.attributes["element_width"] == 64
    assert copy.attributes["ordered_overlap_semantics"] is True
    assert copy.attributes["iterative"] is True
    assert len(copy.args) == 5
    assert {item.attributes.get("machine_register") for item in instructions} >= {
        "RSI", "RDI",
    }


def test_locked_xadd_has_atomic_observation_memory_and_exact_flags():
    lifting = raise_binary_region_to_ssa(
        b"\xf0\x0f\xc1\x11\xc3", maximum_file_size=5, size=5,
        base_address=0x1000, name="locked_xadd",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "XADD_RM32_R32"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        instruction for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    observed = next(
        item for item in instructions if item.op == "AtomicExchangeAddObserved"
    )
    memory = next(
        item for item in instructions if item.op == "AtomicExchangeAddMemory"
    )
    assert observed.attributes["ordering"] == "sequentially-consistent"
    assert memory.attributes["locked"] is True
    assert {item.attributes.get("machine_flag") for item in instructions} >= {
        "CF", "PF", "AF", "ZF", "SF", "OF",
    }


def test_btc_complements_selected_memory_bit_and_exports_prior_cf():
    lifting = raise_binary_region_to_ssa(
        b"\x0f\xba\x39\x05\xc3", maximum_file_size=5, size=5,
        base_address=0x1000, name="btc",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "BTC_RM32_IMM8"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        instruction for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    assert any(item.attributes.get("machine_bit_complement") for item in instructions)
    assert any(item.op == "Store" and item.attributes.get("width") == 32 for item in instructions)
    assert any(item.attributes.get("machine_flag") == "CF" for item in instructions)


@pytest.mark.parametrize(("encoded", "amount"), (
    (b"\x48\x83\x29\x01\xc3", 1),
    (b"\x48\x81\x29\x01\x00\x00\x00\xc3", 1),
))
def test_sub_memory_destination_decodes_executes_and_legalizes(encoded, amount):
    lifting = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="sub_memory",
        full_vocabulary_report=True,
    )
    instruction = lifting.decoded[0]
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2000, 8).write_unsigned(
        0x2000, 64, 7,
    )
    state = MachineExecutionState(
        pc=0x1000,
        registers=(0, 0x2000, *(0 for _ in range(14))),
        memory=memory,
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )

    executed = executor.step(state)

    assert instruction.token.name in {"SUB_R64_IMM8", "SUB_R64_IMM32"}
    assert executed.state.memory.read_unsigned(0x2000, 64) == 7 - amount
    assert lifting.failed_vocabulary == ()
    assert lifting.function is not None
    assert "dialect" not in lifting.function.metadata


def test_sar_imm8_and_jns_rel32_complete_decode_vm_and_ssa_path():
    encoded = b"\x48\xc1\xf8\x05\x0f\x89\x00\x00\x00\x00\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded,
        maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="sar_jns",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    state = MachineExecutionState(
        pc=0x1000,
        registers=(((-64) & ((1 << 64) - 1)), *(0 for _ in range(15))),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )

    shifted = executor.step(state)

    assert [item.token.name for item in lifting.decoded] == [
        "SAR_RM64_IMM8", "JNS_REL32", "RET_NEAR",
    ]
    assert shifted.state.registers[0] == ((-2) & ((1 << 64) - 1))
    assert lifting.failed_vocabulary == ()
    assert any(
        item.op == "AShr"
        for block in lifting.function.blocks.values()
        for item in block.instrs
    )


def test_movsx_r64_rm8_sib_decodes_executes_and_legalizes():
    # movsx rdi, byte ptr [r14 + rbp + 0x20]; ret
    encoded = b"\x49\x0f\xbe\x7c\x2e\x20\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="movsx_rm8_sib",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2020, 1).write_unsigned(
        0x2020, 8, 0x80,
    )
    registers = [0] * 16
    registers[14] = 0x2000
    registers[5] = 0
    state = MachineExecutionState(
        pc=0x1000, registers=tuple(registers), memory=memory,
    )
    executed = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    ).step(state)

    assert lifting.decoded[0].token.name == "MOVSX_R64_RM8"
    assert executed.state.registers[7] == 0xFFFFFFFFFFFFFF80
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("left", "right", "overflow"), (
    (3, -4, False),
    (1 << 62, 4, True),
))
def test_imul_r64_rm64_decodes_and_has_exact_vm_overflow_flags(
    left, right, overflow,
):
    # imul r8, qword ptr [rax + 0x28]; ret
    encoded = b"\x4c\x0f\xaf\x40\x28\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="imul_rm64",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2028, 8).write_unsigned(
        0x2028, 64, right & ((1 << 64) - 1),
    )
    registers = [0] * 16
    registers[0] = 0x2000
    registers[8] = left & ((1 << 64) - 1)
    state = MachineExecutionState(
        pc=0x1000, registers=tuple(registers), memory=memory,
    )
    executed = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    ).step(state)

    assert lifting.decoded[0].token.name == "IMUL_R64_RM64"
    assert executed.state.registers[8] == (left * right) & ((1 << 64) - 1)
    assert bool(executed.state.flags & 1) is overflow
    assert bool(executed.state.flags & (1 << 11)) is overflow
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    lowered = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    assert any(item.op == "SMulLow" for item in lowered)
    assert any(item.op == "SMulOverflow" for item in lowered)


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x80\xca\x80\xc3", "OR_RM8_IMM8"),
    (b"\x81\xe9\x87\x00\x00\x00\xc3", "SUB_RM32_IMM32"),
    (b"\x48\x83\xf2\x01\xc3", "XOR_RM64_IMM8"),
))
def test_cached_group_immediate_tail_decodes_and_legalizes(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x0f\x4f\xc7\xc3", "CMOVG_R32_RM32"),
    (b"\x48\x0f\x4f\xf0\xc3", "CMOVG_R64_RM64"),
    (b"\x45\x0f\x49\xdf\xc3", "CMOVNS_R32_RM32"),
))
def test_cached_conditional_move_tail_legalizes_to_repository_ssa(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert any(
        item.op == "Select"
        for block in lifting.function.blocks.values()
        for item in block.instrs
    )


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x05\x81\x00\x00\x00\xc3", "ADD_EAX_IMM32"),
    (b"\x2d\xb8\x01\x00\x00\xc3", "SUB_EAX_IMM32"),
    (b"\x0c\xe0\xc3", "OR_AL_IMM8"),
    (b"\x80\xeb\x01\xc3", "SUB_RM8_IMM8"),
))
def test_cached_accumulator_and_byte_tail_legalizes(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("encoded", "token", "ordinary"), (
    (b"\x66\x41\x0b\xc8\xc3", "OR_R16_RM16", True),
    (b"\x4c\x29\x6d\x00\xc3", "SUB_RM64_R64", True),
    (b"\x49\x0f\x4e\xcc\xc3", "CMOVLE_R64_RM64", True),
    (b"\x41\x0f\x93\xc0\xc3", "SETAE_RM8", True),
    (b"\x87\x43\x18\xc3", "XCHG_RM32_R32", True),
))
def test_cached_final_integer_decode_tail(encoded, token, ordinary):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True,
    )

    assert lifting.decoded[0].token.name == token
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    assert ("dialect" not in lifting.function.metadata) is ordinary


def test_cdq_idiv_rm32_decodes_and_executes_exact_signed_division():
    encoded = b"\x99\x41\xf7\xf8\xc3"  # cdq; idiv r8d; ret
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cdq_idiv32",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    registers = [0] * 16
    registers[0] = (-17) & 0xFFFFFFFF
    registers[8] = 5
    state = MachineExecutionState(pc=0x1000, registers=tuple(registers))
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )

    extended = executor.step(state).state
    divided = executor.step(extended).state

    assert [item.token.name for item in lifting.decoded] == [
        "CDQ", "IDIV_RM32", "RET_NEAR",
    ]
    assert extended.registers[2] == 0xFFFFFFFF
    assert divided.registers[0] == ((-3) & 0xFFFFFFFF)
    assert divided.registers[2] == ((-2) & 0xFFFFFFFF)
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    assert any(item.attributes.get("machine_cdq_high_half") for item in instructions)
    assert any(item.op == "WideDivCheck" for item in instructions)


@pytest.mark.parametrize(("source", "prior", "expected", "zero"), (
    (0x100, 19, 8, False),
    (0, 19, 19, True),
))
def test_bsr_rm32_tracks_zero_undefinedness_in_machine_state(
    source, prior, expected, zero,
):
    encoded = b"\x41\x0f\xbd\x4c\x82\x14\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="bsr32",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    memory = PagedByteMemory.empty().map_zeroes(0x2014, 4).write_unsigned(
        0x2014, 32, source,
    )
    registers = [0] * 16
    registers[10] = 0x2000
    registers[0] = 0
    registers[1] = prior
    state = MachineExecutionState(
        pc=0x1000, registers=tuple(registers), memory=memory,
    )
    executed = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    ).step(state).state

    assert lifting.decoded[0].token.name == "BSR_R32_RM32"
    assert executed.registers[1] == expected
    assert bool(executed.flags & (1 << 6)) is zero
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    assert any(item.op == "MsbIndex" for item in instructions)
    assert any(
        item.attributes.get("machine_undefined_destination") == "preserve-prior"
        for item in instructions
    )


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\xf2\x48\x0f\x2a\xcb\xc3", "CVTSI2SD_XMM_RM64"),
    (b"\x66\x49\x0f\x6e\xc7\xc3", "MOVQ_XMM_RM64"),
))
def test_final_cached_vector_transfer_forms_decode(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True,
    )
    assert lifting.decoded[0].token.name == token
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    expected = (
        "SInt64ToFloat64Bits"
        if token == "CVTSI2SD_XMM_RM64" else None
    )
    if expected is not None:
        assert any(
            item.op == expected
            for block in lifting.function.blocks.values()
            for item in block.instrs
        )


def test_scalar_conversion_preserves_upper_lane_and_movq_clears_it():
    encoded = (
        b"\xf2\x48\x0f\x2a\xcb"  # cvtsi2sd xmm1, rbx
        b"\x66\x49\x0f\x6e\xc7"  # movq xmm0, r15
        b"\xc3"
    )
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="scalar_vector_transfers",
        full_vocabulary_report=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    registers = [0] * 16
    registers[3] = (-3) & ((1 << 64) - 1)
    registers[15] = 0xFEDCBA9876543210
    vectors = [0] * 16
    vectors[0] = (0xAAAAAAAAAAAAAAAA << 64) | 1
    vectors[1] = (0xBBBBBBBBBBBBBBBB << 64) | 2
    state = MachineExecutionState(
        pc=0x1000, registers=tuple(registers),
        vector_registers=tuple(vectors),
    )
    executor = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    )

    converted = executor.step(state).state
    moved = executor.step(converted).state
    double_bits = int.from_bytes(struct.pack("<d", -3.0), "little")

    assert converted.vector_registers[1] == (
        (0xBBBBBBBBBBBBBBBB << 64) | double_bits
    )
    assert moved.vector_registers[0] == 0xFEDCBA9876543210


def test_movsx_r32_rm8_extended_sib_legalizes():
    encoded = b"\x42\x0f\xbe\x84\x0a\x0b\x02\x00\x00\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="movsx_r32_rm8_sib",
        full_vocabulary_report=True,
    )

    assert lifting.decoded[0].token.name == "MOVSX_R32_RM8"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


def test_cfg_decoder_follows_branch_targets_and_retains_unreachable_bytes():
    # test eax,eax; je alternate; mov eax,1; ret; padding; alternate: mov eax,2; ret
    encoded = (
        b"\x85\xc0\x74\x08\xb8\x01\x00\x00\x00\xc3"
        b"\xcc\xcc\xb8\x02\x00\x00\x00\xc3"
    )
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cfg_with_padding",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert [item.address for item in lifting.decoded] == [
        0x1000, 0x1002, 0x1004, 0x1009, 0x100C, 0x1011,
    ]
    assert lifting.function.metadata["machine_unreachable_spans"] == (
        (0x100A, 0x100C),
    )
    assert lifting.function.metadata["machine_unreachable_byte_count"] == 2


def test_cfg_decoder_rejects_branch_into_an_instruction_body():
    # The conditional target 0x1005 enters the immediate of mov eax,imm32.
    encoded = b"\x85\xc0\x74\x01\xb8\x01\x00\x00\x00\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="overlapping_target",
        cfg_decode=True,
    )

    assert lifting.function is None
    assert [item.category for item in lifting.failed_vocabulary] == [
        "overlapping_control_target",
    ]


def test_cfg_decoder_retains_pdata_end_fallthrough_as_external_control():
    encoded = b"\x90"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cross_pdata_fallthrough",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert lifting.function is not None
    branch = next(
        item
        for block in lifting.function.blocks.values()
        for item in block.instrs
        if item.op == "Br" and item.attributes.get("target_address") is not None
    )
    assert branch.attributes["target_address"] == 0x1001
    assert branch.attributes["machine_control_transfer"] == (
        "cross-region-fallthrough"
    )
    assert lifting.function.metadata["machine_external_control_targets"] == (
        0x1001,
    )


def test_ordinary_ssa_preserves_conditional_edge_outside_pdata_region():
    # test eax,eax; jne outside; ret fallthrough
    encoded = b"\x85\xc0\x75\x7f\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="external_conditional",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    branch = next(
        item
        for block in lifting.function.blocks.values()
        for item in block.instrs
        if item.op == "CondBr"
    )
    assert branch.attributes["true_target"] is None
    assert branch.attributes["true_target_address"] == 0x1083
    assert branch.attributes["false_target"] == "block_0000000000001004"
    assert branch.attributes["machine_external_control"] is True
    assert lifting.function.metadata["machine_external_control_targets"] == (
        0x1083,
    )
    assert lifting.function.metadata["requires_machine_address_linking"] is True


def test_cyclic_ordinary_ssa_preserves_conditional_edge_outside_pdata_region():
    # test eax,eax; je outside; dec eax; jne back to test; ret
    encoded = b"\x85\xc0\x74\x7f\xff\xc8\x75\xf8\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cyclic_external_conditional",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    external = next(
        item
        for block in lifting.function.blocks.values()
        for item in block.instrs
        if (
            item.op == "CondBr"
            and item.attributes.get("true_target_address") == 0x1083
        )
    )
    assert external.attributes["true_target"] is None
    assert external.attributes["false_target"] == "block_0000000000001004"
    assert external.attributes["machine_external_control"] is True
    assert lifting.function.metadata["machine_external_control_targets"] == (
        0x1083,
    )
    assert lifting.function.metadata["requires_machine_address_linking"] is True


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x48\xff\xc0\xc3", "INC_RM64"),
    (b"\xff\xc8\xc3", "DEC_RM32"),
    (b"\xf7\xd8\xc3", "NEG_RM32"),
    (b"\xf6\xd8\xc3", "NEG_RM8"),
))
def test_unary_arithmetic_family_legalizes_with_explicit_flags(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    flags = {
        item.attributes.get("machine_flag")
        for block in lifting.function.blocks.values()
        for item in block.instrs
        if item.attributes.get("machine_flag")
    }
    assert {"CF", "PF", "AF", "ZF", "SF", "OF"} <= flags


@pytest.mark.parametrize(("encoded", "token", "writes_result"), (
    (b"\x0f\xba\xe0\x07\xc3", "BT_RM32_IMM8", False),
    (b"\x0f\xba\xf0\x07\xc3", "BTR_RM32_IMM8", True),
))
def test_register_immediate_bit_test_family_legalizes(
    encoded, token, writes_result,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = [
        item for block in lifting.function.blocks.values() for item in block.instrs
    ]
    assert any(item.attributes.get("machine_flag") == "CF" for item in instructions)
    assert any("machine_bit_reset" in item.attributes for item in instructions) is writes_result


@pytest.mark.parametrize(("encoded", "token", "mutation"), (
    (b"\x48\x0f\xa3\xca\xc3", "BT_RM64_R64", None),
    (b"\x41\x0f\xab\xc3\xc3", "BTS_RM32_R32", "set"),
    (b"\x0f\xba\x2f\x18\xc3", "BTS_RM32_IMM8", "set"),
    (b"\x0f\xba\x73\x10\x13\xc3", "BTR_RM32_IMM8", "reset"),
))
def test_dynamic_register_and_constant_memory_bit_tests_legalize(
    encoded, token, mutation,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = [
        item for block in lifting.function.blocks.values() for item in block.instrs
    ]
    assert any(item.attributes.get("machine_bit_test_index") for item in instructions)
    if mutation == "set":
        assert any(item.attributes.get("machine_bit_set") for item in instructions)
    elif mutation == "reset":
        assert any(item.attributes.get("machine_bit_reset") for item in instructions)


def test_shr_rm64_cl_decodes_and_executes_zero_count_preservation():
    encoded = b"\x48\xd3\xe8\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="shr_rm64_cl",
        full_vocabulary_report=True, cfg_decode=True,
    )
    program = SimpleNamespace(
        image=SimpleNamespace(image_base=0x1000, entrypoint_rva=0),
        functions=(SimpleNamespace(report=SimpleNamespace(
            instructions=lifting.decoded,
        )),),
    )
    registers = [0] * 16
    registers[0] = 0x8000000000000001
    registers[1] = 0
    state = MachineExecutionState(
        pc=0x1000, registers=tuple(registers), flags=0x8D5,
    )
    executed = MachineExecutionOrchestrator(
        program, effect_handlers=default_effect_handlers(),
    ).step(state).state

    assert lifting.decoded[0].token.name == "SHR_RM64_CL"
    assert executed.registers[0] == state.registers[0]
    assert executed.flags == state.flags
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = [
        item
        for block in lifting.function.blocks.values()
        for item in block.instrs
    ]
    assert any(
        item.attributes.get("machine_zero_shift_preserves_destination")
        for item in instructions
    )
    assert {
        item.attributes.get("machine_zero_shift_preserves_flag")
        for item in instructions
    } >= {"CF", "PF", "AF", "ZF", "SF", "OF"}


def test_shr_rm32_cl_decodes_into_exact_machine_state_dialect():
    encoded = b"\xd3\xe8\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="shr_rm32_cl",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "SHR_RM32_CL"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


def test_sar_rm32_imm8_legalizes_with_explicit_shift_state():
    encoded = b"\xc1\xf8\x03\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="sar_rm32_imm8",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "SAR_RM32_IMM8"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x66\x25\x00\xff\xc3", "AND_AX_IMM16"),
    (b"\x66\xc7\x47\x08\x03\x03\xc3", "MOV_RM16_IMM16"),
    (b"\x48\x81\xf2\xb3\x4d\x5b\x05\xc3", "XOR_RM64_IMM32"),
))
def test_remaining_integer_width_forms_legalize(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


def test_bsr_r64_rm64_legalizes_with_explicit_zero_source_preservation():
    encoded = b"\x48\x0f\xbd\xd1\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="bsr_r64_rm64",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "BSR_R64_RM64"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    index = next(item for item in instructions if item.op == "MsbIndex")
    selected = next(
        item for item in instructions
        if item.attributes.get("machine_undefined_destination") == "preserve-prior"
    )
    zero = next(
        item for item in instructions
        if item.attributes.get("machine_flag") == "ZF"
    )
    assert index.attributes["width"] == 64
    assert index.attributes["zero_totalized"] is True
    assert index.attributes["machine_bit_scan_reverse"] is True
    assert selected.args[0] == zero.res


@pytest.mark.parametrize(("encoded", "token", "contract"), (
    (b"\x0f\x10\xc1\xc3", "MOVUPS_XMM_XMMM128", "full"),
    (b"\x66\x0f\x6f\xc1\xc3", "MOVDQA_XMM_XMMM128", "full"),
    (b"\x0f\x29\xc1\xc3", "MOVAPS_XMMM128_XMM", "full"),
    (b"\x0f\x57\xc1\xc3", "XORPS_XMM_XMMM128", "xor"),
    (b"\xf2\x0f\x10\xc1\xc3", "MOVSD_XMM_XMMM64", "preserve-upper"),
    (b"\xf3\x0f\x7e\xc1\xc3", "MOVQ_XMM_XMMM64", "zero-upper"),
    (b"\x66\x49\x0f\x6e\xc7\xc3", "MOVQ_XMM_RM64", "zero-upper"),
))
def test_xmm_bit_pattern_families_are_explicit_repository_ssa(
    encoded, token, contract,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert any(argument.dtype == "int128" for argument in lifting.function.args)
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    if contract == "xor":
        assert any(item.attributes.get("machine_vector_bit_pattern") for item in instructions)
    elif contract == "preserve-upper":
        assert any(item.attributes.get("machine_vector_preserve_upper") for item in instructions)
    elif contract == "zero-upper":
        assert any(item.attributes.get("machine_vector_zero_upper") == 64 for item in instructions)


def test_looping_xmm_state_receives_vector_register_phis():
    # xorps xmm0,xmm1; test eax,eax; jne back to xorps; ret
    encoded = b"\x0f\x57\xc1\x85\xc0\x75\xf9\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="xmm_loop",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert lifting.function.metadata["machine_loop_state_phis"] is True
    header = lifting.function.blocks["entry"]
    vectors = [
        item for item in header.instrs
        if item.op == "Phi"
        and item.attributes.get("machine_state") == "vector-register"
    ]
    assert {item.attributes["machine_register"] for item in vectors} == {
        "XMM0", "XMM1",
    }
    assert all(item.res.dtype == "int128" and len(item.args) == 2 for item in vectors)


@pytest.mark.parametrize(("encoded", "token", "lane_width"), (
    (b"\x66\x0f\x60\xc8\xc3", "PUNPCKLBW_XMM_XMMM128", 8),
    (b"\x66\x0f\x61\xc8\xc3", "PUNPCKLWD_XMM_XMMM128", 16),
    (b"\x66\x0f\x6c\xc8\xc3", "PUNPCKLQDQ_XMM_XMMM128", 64),
))
def test_unpack_low_is_one_lane_parameterized_repository_operation(
    encoded, token, lane_width,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    unpack = next(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "VectorUnpackLow"
    )
    assert unpack.res.dtype == "int128"
    assert unpack.attributes["lane_width"] == lane_width
    assert unpack.attributes["vector_width"] == 128


def test_packed_qword_add_is_lane_modular_repository_operation():
    encoded = b"\x66\x0f\xd4\xc8\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="paddq",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "PADDQ_XMM_XMMM128"
    assert lifting.failed_vocabulary == ()
    add = next(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "VectorAddModulo"
    )
    assert add.attributes["lane_width"] == 64
    assert add.attributes["vector_width"] == 128
    assert add.attributes["machine_vector_bit_pattern"] is True


@pytest.mark.parametrize(("encoded", "token", "ordered"), (
    (b"\x66\x0f\x2e\xc1\xc3", "UCOMISD_XMM_XMMM64", False),
    (b"\x66\x0f\x2f\xc1\xc3", "COMISD_XMM_XMMM64", True),
))
def test_scalar_float_compare_carries_explicit_mxcsr_and_bit_predicates(
    encoded, token, ordered,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    names = lifting.function.metadata["argument_names"]
    assert names.count("__machine_mxcsr") == 1
    mxcsr_argument = lifting.function.args[names.index("__machine_mxcsr")]
    assert mxcsr_argument.dtype == "int32"
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    assert sum(item.op == "Float64IsNaNBits" for item in instructions) == 2
    assert sum(
        item.op == "Float64IsSignalingNaNBits" for item in instructions
    ) == (0 if ordered else 2)
    invalid = next(item for item in instructions if item.op == "MXCSRInvalid")
    assert invalid.args[0] == mxcsr_argument
    assert invalid.attributes["may_trap"] is True
    assert invalid.attributes["status_bit"] == 0
    assert invalid.attributes["mask_bit"] == 7
    assert {item.attributes.get("machine_flag") for item in instructions} >= {
        "CF", "PF", "ZF",
    }


def test_looping_float_compare_carries_mxcsr_phi():
    # ucomisd xmm0,xmm1; jne back to compare; ret
    encoded = b"\x66\x0f\x2e\xc1\x75\xfa\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="ucomisd_loop",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    header = lifting.function.blocks["entry"]
    mxcsr = [
        item for item in header.instrs
        if item.op == "Phi" and item.attributes.get("machine_state") == "mxcsr"
    ]
    assert len(mxcsr) == 1
    assert mxcsr[0].res.dtype == "int32"
    assert len(mxcsr[0].args) == 2


def test_cvtsi2sd_legalizes_with_explicit_rounding_and_precision_state():
    encoded = b"\xf2\x48\x0f\x2a\xcb\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cvtsi2sd",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    names = lifting.function.metadata["argument_names"]
    assert names.count("__machine_mxcsr") == 1
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    conversion = next(
        item for item in instructions if item.op == "SInt64ToFloat64Bits"
    )
    precision = next(item for item in instructions if item.op == "MXCSRPrecision")
    assert conversion.attributes["rounding_source"] == "mxcsr"
    assert conversion.attributes["integer_only_encoding"] is True
    assert precision.attributes["may_trap"] is True
    assert precision.attributes["status_bit"] == 5
    assert precision.attributes["mask_bit"] == 12
    assert any(
        item.attributes.get("machine_vector_preserve_upper")
        for item in instructions
    )


@pytest.mark.parametrize(("encoded", "token", "dialect"), (
    (b"\x40\xfe\xce\xc3", "DEC_RM8", None),
    (b"\xfe\xc2\xc3", "INC_RM8", None),
    (b"\x45\x2a\xd0\xc3", "SUB_R8_RM8", None),
    (b"\x66\x0f\x2e\xce\xc3", "UCOMISD_XMM_XMMM64", None),
    (b"\xf3\x0f\x7e\x0a\xc3", "MOVQ_XMM_XMMM64", None),
    (b"\xf2\x0f\x58\xc9\xc3", "ADDSD_XMM_XMMM64", None),
    (b"\x66\x0f\x6c\xc8\xc3", "PUNPCKLQDQ_XMM_XMMM128", None),
))
def test_final_cached_decode_tail_has_exact_semantics(encoded, token, dialect):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    if dialect is None:
        assert lifting.failed_vocabulary == ()
        assert "dialect" not in lifting.function.metadata
    else:
        assert lifting.function.metadata["dialect"] == dialect


def test_addsd_uses_encoded_binary64_result_and_ordered_mxcsr_transition():
    encoded = b"\xf2\x0f\x58\xc9\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="addsd_exact",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    result = next(item for item in instructions if item.op == "Float64AddBits")
    status = next(item for item in instructions if item.op == "MXCSRFloat64Add")
    assert result.attributes["host_float_independent"] is True
    assert result.attributes["rounding_source"] == "mxcsr"
    assert status.attributes["may_trap"] is True
    assert status.attributes["trap_before_destination_write"] is True
    assert status.attributes["exception_status_bits"] == (0, 1, 2, 3, 4, 5)
    assert status.attributes["exception_mask_bits"] == (7, 8, 9, 10, 11, 12)
    assert status.args[1:] == result.args[:2]
    assert status.args[0] == result.args[2]
    assert any(
        item.attributes.get("machine_vector_preserve_upper")
        for item in instructions
    )


def test_mulsd_uses_encoded_binary64_result_and_ordered_mxcsr_transition():
    encoded = b"\xf2\x0f\x59\xc9\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="mulsd_exact",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "MULSD_XMM_XMMM64"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    result = next(
        item for item in instructions if item.op == "Float64MultiplyBits"
    )
    status = next(
        item for item in instructions if item.op == "MXCSRFloat64Multiply"
    )
    assert result.attributes["host_float_independent"] is True
    assert result.attributes["rounding_source"] == "mxcsr"
    assert status.attributes["may_trap"] is True
    assert status.attributes["trap_before_destination_write"] is True
    assert status.args[1:] == result.args[:2]
    assert status.args[0] == result.args[2]


@pytest.mark.parametrize(("encoded", "token", "dialect"), (
    (b"\xc0\xe1\x03\xc3", "SHL_RM8_IMM8", None),
    (b"\x41\x80\xc2\xff\xc3", "ADD_RM8_IMM8", None),
    (b"\x7a\x00\xc3", "JP_REL8", None),
    (b"\x66\x0f\x60\xc8\xc3", "PUNPCKLBW_XMM_XMMM128", None),
    (b"\xf2\x0f\x59\xc1\xc3", "MULSD_XMM_XMMM64", None),
    (b"\x66\x0f\xd4\xc8\xc3", "PADDQ_XMM_XMMM128", None),
))
def test_next_cached_decode_tail_has_exact_semantics(encoded, token, dialect):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    if dialect is None:
        assert lifting.failed_vocabulary == ()
        assert "dialect" not in lifting.function.metadata
    else:
        assert lifting.function.metadata["dialect"] == dialect


@pytest.mark.parametrize(("encoded", "token", "dialect"), (
    (b"\x48\x69\xc8\x4d\xef\xe8\x72\xc3", "IMUL_R64_RM64_IMM32", None),
    (b"\x66\x0f\x2f\xf0\xc3", "COMISD_XMM_XMMM64", None),
    (b"\x66\x0f\x61\xc8\xc3", "PUNPCKLWD_XMM_XMMM128", None),
))
def test_third_cached_decode_tail_has_expected_lowering(encoded, token, dialect):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert not any(item.category == "decode" for item in lifting.failed_vocabulary)
    if dialect is None:
        assert lifting.failed_vocabulary == ()
        assert "dialect" not in lifting.function.metadata
        expected = (
            "SMulOverflow"
            if token.startswith("IMUL_") else (
                "VectorUnpackLow"
                if token.startswith("PUNPCKL") else "MXCSRInvalid"
            )
        )
        assert any(
            item.op == expected
            for block in lifting.function.blocks.values()
            for item in block.instrs
        )
    else:
        assert lifting.function.metadata["dialect"] == dialect


def test_movd_xmm_rm32_legalizes_with_zero_upper_contract():
    encoded = b"\x66\x0f\x6e\x0a\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="movd_xmm_rm32",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "MOVD_XMM_RM32"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert any(
        item.attributes.get("machine_vector_zero_upper") == 32
        for block in lifting.function.blocks.values()
        for item in block.instrs
    )


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x48\x1b\xc1\xc3", "SBB_R64_RM64"),
    (b"\x1b\xc1\xc3", "SBB_R32_RM32"),
    (b"\x48\x98\xc3", "CDQE"),
    (b"\x48\x99\xc3", "CQO"),
    (b"\x99\xc3", "CDQ"),
    (b"\xf7\xd0\xc3", "NOT_RM32"),
))
def test_safe_integer_machine_families_legalize_to_repository_ssa(
    encoded, token,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("encoded", "token", "kind", "vector"), (
    (b"\xcc", "INT3", "breakpoint", 3),
    (b"\xcd\x2d", "INT_IMM8", "software-interrupt", 0x2D),
))
def test_architectural_interrupts_lower_to_explicit_nonreturning_traps(
    encoded, token, kind, vector,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    trap = next(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Trap"
    )
    assert trap.res is None
    assert trap.attributes["trap_kind"] == kind
    assert trap.attributes["interrupt_vector"] == vector
    assert trap.attributes["non_returning"] is True


@pytest.mark.parametrize(("encoded", "source_kind"), (
    (b"\x48\xff\xe0", "register"),
    (b"\x48\xff\x25\x10\x00\x00\x00", "memory"),
))
def test_indirect_jump_carries_computed_target_and_complete_machine_state(
    encoded, source_kind,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=f"indirect_{source_kind}",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == "JMP_RM64"
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert lifting.function.metadata["requires_dynamic_target_linking"] is True
    branch = next(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
        if instruction.op == "IndirectBr"
    )
    assert branch.res is None
    assert branch.attributes["target_source"] == source_kind
    assert branch.attributes["indirect_operand"] == (
        "register" if source_kind == "register" else "rip-relative-memory"
    )
    assert branch.attributes["indirect_slot_address"] == (
        None if source_kind == "register" else 0x1017
    )
    assert branch.attributes["complete_machine_state"] is True
    layout = branch.attributes["state_layout"]
    assert layout[0:2] == ("target", "memory")
    assert {f"register.{name}" for name in (
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    )} <= set(layout)
    assert {f"vector-register.xmm{index}" for index in range(16)} <= set(layout)
    assert "system.amd64.mxcsr" in layout
    assert {f"flag.{name}" for name in (
        "cf", "pf", "af", "zf", "sf", "of", "df",
    )} <= set(layout)
    assert len(branch.args) == len(layout)
    owner = next(
        block for block in lifting.function.blocks.values()
        if branch in block.instrs
    )
    assert owner.instrs[-1] is branch
    assert owner.successors == []


@pytest.mark.parametrize(("encoded", "token"), (
    (b"\x0f\x9c\xc0\xc3", "SETL_RM8"),
    (b"\x0f\x98\xc3\xc3", "SETS_RM8"),
    (b"\x48\x0f\x49\xca\xc3", "CMOVNS_R64_RM64"),
    (b"\x48\x0f\x47\xca\xc3", "CMOVA_R64_RM64"),
    (b"\x48\x0f\x4d\xc8\xc3", "CMOVGE_R64_RM64"),
    (b"\x48\x0f\x46\xc7\xc3", "CMOVBE_R64_RM64"),
))
def test_remaining_condition_variants_legalize(encoded, token):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata


@pytest.mark.parametrize(("encoded", "token", "count", "direction"), (
    (b"\x48\xc1\xc0\x01\xc3", "ROL_RM64_IMM8", 1, "left"),
    (b"\x48\xc1\xc0\x09\xc3", "ROL_RM64_IMM8", 9, "left"),
    (b"\x48\xc1\xc0\x40\xc3", "ROL_RM64_IMM8", 64, "left"),
    (b"\x48\xc1\xc8\x01\xc3", "ROR_RM64_IMM8", 1, "right"),
    (b"\x48\xc1\xc8\x09\xc3", "ROR_RM64_IMM8", 9, "right"),
))
def test_rotate_immediate_legalizes_without_changing_unrelated_flags(
    encoded, token, count, direction,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=f"rol_{count}",
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    if count % 64 == 0:
        assert not any(item.attributes.get("machine_rotate") for item in instructions)
    else:
        assert any(item.attributes.get("machine_rotate") == direction for item in instructions)
        assert any(item.attributes.get("machine_flag") == "CF" for item in instructions)
        assert any(item.attributes.get("machine_flag") == "OF" for item in instructions) == (
            count % 64 == 1
        )


@pytest.mark.parametrize(("encoded", "token", "width"), (
    (b"\x48\x0f\xaf\xc1\xc3", "IMUL_R64_RM64", 64),
    (b"\x0f\xaf\xc1\xc3", "IMUL_R32_RM32", 32),
    (b"\x48\x6b\xc1\xff\xc3", "IMUL_R64_RM64_IMM8", 64),
    (b"\x48\x69\xc1\x00\x00\x00\x80\xc3", "IMUL_R64_RM64_IMM32", 64),
))
def test_signed_multiply_uses_exact_fixed_width_repository_primitives(
    encoded, token, width,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    low = next(item for item in instructions if item.op == "SMulLow")
    overflow = next(item for item in instructions if item.op == "SMulOverflow")
    assert low.attributes["width"] == width
    assert overflow.attributes["width"] == width
    assert overflow.attributes["machine_flag"] == "CF/OF"
    assert low.args == overflow.args
    assert not any(
        item.attributes.get("machine_flag") in {"PF", "AF", "ZF", "SF"}
        for item in instructions
    )


@pytest.mark.parametrize(("encoded", "token", "width"), (
    (b"\x48\xf7\xe1\xc3", "MUL_RM64", 64),
    (b"\xf7\xe1\xc3", "MUL_RM32", 32),
))
def test_unsigned_accumulator_multiply_exposes_both_product_halves(
    encoded, token, width,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    low = next(item for item in instructions if item.op == "UMulLow")
    high = next(item for item in instructions if item.op == "UMulHigh")
    overflow = next(
        item for item in instructions
        if item.op == "Ne" and item.attributes.get("machine_flag") == "CF/OF"
    )
    assert low.attributes["width"] == high.attributes["width"] == width
    assert low.args == high.args
    assert overflow.args[0] == high.res
    assert not any(
        item.attributes.get("machine_flag") in {"PF", "AF", "ZF", "SF"}
        for item in instructions
    )


def test_signed_accumulator_multiply_exposes_both_product_halves():
    encoded = b"\x48\xf7\xe9\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="imul_accumulator",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "IMUL_RM64"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    low = next(item for item in instructions if item.op == "SMulLow")
    high = next(item for item in instructions if item.op == "SMulHigh")
    overflow = next(item for item in instructions if item.op == "SMulOverflow")
    assert low.args == high.args == overflow.args
    assert overflow.attributes["machine_flag"] == "CF/OF"


def test_cvtsi2ss_uses_integer_only_binary32_and_mxcsr_precision_contract():
    encoded = b"\xf3\x48\x0f\x2a\xc1\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cvtsi2ss",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "CVTSI2SS_XMM_RM64"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    converted = next(item for item in instructions if item.op == "SInt64ToFloat32Bits")
    precision = next(item for item in instructions if item.op == "MXCSRPrecision")
    assert converted.attributes["target_format"] == "binary32"
    assert converted.attributes["integer_only_encoding"] is True
    assert precision.attributes["exact_magnitude_bits"] == 24
    assert any(
        item.attributes.get("machine_vector_preserve_upper")
        for item in instructions
    )


def test_addss_lowers_encoded_binary32_result_and_mxcsr_transition():
    encoded = b"\xf3\x0f\x58\xc0\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="addss",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "ADDSS_XMM_XMMM32"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    result = next(item for item in instructions if item.op == "Float32AddBits")
    status = next(item for item in instructions if item.op == "MXCSRFloat32Add")
    assert result.attributes["format"] == "ieee754-binary32"
    assert result.attributes["host_float_independent"] is True
    assert status.attributes["trap_before_destination_write"] is True
    assert any(
        item.attributes.get("machine_vector_preserve_upper")
        for item in instructions
    )


def test_divss_lowers_encoded_binary32_result_and_ordered_mxcsr_transition():
    encoded = b"\xf3\x0f\x5e\xc1\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="divss",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "DIVSS_XMM_XMMM32"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    result = next(item for item in instructions if item.op == "Float32DivideBits")
    status = next(item for item in instructions if item.op == "MXCSRFloat32Divide")
    assert result.attributes["format"] == "ieee754-binary32"
    assert result.attributes["host_float_independent"] is True
    assert status.attributes["exception_order"][2] == "divide-by-zero"
    assert status.attributes["trap_before_destination_write"] is True


def test_comiss_lowers_encoded_binary32_ordered_comparison_and_flags():
    encoded = b"\x0f\x2f\xc1\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="comiss",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "COMISS_XMM_XMMM32"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    assert any(item.op == "Float32IsNaNBits" for item in instructions)
    assert any(item.op == "Float32BitsLt" for item in instructions)
    assert any(item.op == "Float32BitsEq" for item in instructions)
    invalid = next(item for item in instructions if item.op == "MXCSRInvalid")
    assert invalid.attributes["may_trap"] is True
    assert {item.attributes.get("machine_flag") for item in instructions} >= {
        "CF", "PF", "ZF",
    }


@pytest.mark.parametrize(("encoded", "token", "required_op"), (
    (b"\x48\x0f\x48\xca\xc3", "CMOVS_R64_RM64", "Select"),
    (b"\xf2\x0f\x5e\xc1\xc3", "DIVSD_XMM_XMMM64", "Float64DivideBits"),
    (b"\x48\x81\xe0\x00\xc0\xff\xff\xc3", "AND_RM64_IMM32", "And"),
    (b"\x0f\xc8\xc3", "BSWAP_R32", "ByteSwap"),
    (b"\x66\xc1\xc9\x08\xc3", "ROR_RM16_IMM8", "Or"),
    (b"\xf2\x0f\x5c\xc1\xc3", "SUBSD_XMM_XMMM64", "Float64SubtractBits"),
    (b"\x0f\x48\xc1\xc3", "CMOVS_R32_RM32", "Select"),
    (b"\xf2\x48\x0f\x2c\xc0\xc3", "CVTTSD2SI_R64_XMMM64", "Float64ToSInt64TruncBits"),
    (b"\x41\x29\x42\x10\xc3", "SUB_RM32_R32", "Sub"),
    (b"\x66\x0f\x38\x29\xc3\xc3", "PCMPEQQ_XMM_XMMM128", "VectorCompareEqualMask"),
    (b"\xd1\xf9\xc3", "SAR_RM32_1", "Shr"),
    (b"\x66\x0f\xfb\xc8\xc3", "PSUBQ_XMM_XMMM128", "VectorSubtractModulo"),
    (b"\x1a\xc0\xc3", "SBB_R8_RM8", "Sub"),
    (b"\x04\x77\xc3", "ADD_AL_IMM8", "Add"),
    (b"\x66\x0f\x73\xd8\x08\xc3", "PSRLDQ_XMM_IMM8", "Shr"),
    (b"\x66\x48\x0f\x7e\xd2\xc3", "MOVQ_RM64_XMM", "And"),
    (b"\x69\xc0\x01\x01\x01\x01\xc3", "IMUL_R32_RM32_IMM32", "SMulLow"),
    (b"\x0f\x54\xc1\xc3", "ANDPS_XMM_XMMM128", "And"),
    (b"\x0f\x4d\xc8\xc3", "CMOVGE_R32_RM32", "Select"),
    (b"\xf7\xea\xc3", "IMUL_RM32", "SMulHigh"),
    (b"\x66\x0f\x70\xc0\x00\xc3", "PSHUFD_XMM_XMMM128_IMM8", "VectorShuffle"),
    (b"\xf2\x0f\x2c\xc7\xc3", "CVTTSD2SI_R32_XMMM64", "Float64ToSInt32TruncBits"),
    (b"\xf3\x0f\xe6\xc0\xc3", "CVTDQ2PD_XMM_XMMM64", "VectorSInt32ToFloat64Bits"),
    (b"\x66\x0f\x7e\xf0\xc3", "MOVD_RM32_XMM", "And"),
))
def test_new_sre_frontier_families_lower_to_repository_ssa(
    encoded, token, required_op,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    assert any(
        item.op == required_op
        for block in lifting.function.blocks.values() for item in block.instrs
    )


def test_locked_cmpxchg32_has_observed_success_memory_and_flag_state():
    encoded = b"\xf0\x0f\xb1\x15\x00\x10\x00\x00\xc3"
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name="cmpxchg32",
        full_vocabulary_report=True, cfg_decode=True,
    )
    assert lifting.decoded[0].token.name == "CMPXCHG_RM32_R32"
    assert lifting.failed_vocabulary == ()
    instructions = tuple(
        item for block in lifting.function.blocks.values() for item in block.instrs
    )
    observed = next(
        item for item in instructions if item.op == "AtomicCompareExchangeObserved"
    )
    success = next(
        item for item in instructions if item.op == "AtomicCompareExchangeSuccess"
    )
    memory = next(
        item for item in instructions if item.op == "AtomicCompareExchangeMemory"
    )
    assert observed.attributes["locked"] is True
    assert success.attributes["machine_flag"] == "ZF"
    assert memory.attributes["ordering"] == "sequentially-consistent"


@pytest.mark.parametrize(("encoded", "token", "signed", "width"), (
    (b"\x48\xf7\xf1\xc3", "DIV_RM64", False, 64),
    (b"\xf7\xf1\xc3", "DIV_RM32", False, 32),
    (b"\x48\xf7\xf9\xc3", "IDIV_RM64", True, 64),
    (b"\xf7\xf9\xc3", "IDIV_RM32", True, 32),
))
def test_accumulator_divide_has_ordered_trap_guard_and_exact_projections(
    encoded, token, signed, width,
):
    lifting = raise_binary_region_to_ssa(
        encoded, maximum_file_size=len(encoded), size=len(encoded),
        base_address=0x1000, name=token.lower(),
        full_vocabulary_report=True, cfg_decode=True,
    )

    assert lifting.decoded[0].token.name == token
    assert lifting.failed_vocabulary == ()
    assert "dialect" not in lifting.function.metadata
    instructions = tuple(
        instruction
        for block in lifting.function.blocks.values()
        for instruction in block.instrs
    )
    guard = next(item for item in instructions if item.op == "WideDivCheck")
    quotient = next(
        item for item in instructions if item.op == "WideDivQuotient"
    )
    remainder = next(
        item for item in instructions if item.op == "WideDivRemainder"
    )
    assert guard.attributes["width"] == width
    assert guard.attributes["signed"] is signed
    assert guard.attributes["may_trap"] is True
    assert guard.attributes["traps"] == ("zero-divisor", "quotient-overflow")
    assert quotient.args[:3] == remainder.args[:3] == guard.args
    assert quotient.args[3] == remainder.args[3] == guard.res
    assert quotient.attributes["signed"] is remainder.attributes["signed"] is signed
    assert not any(item.attributes.get("machine_flag") for item in instructions)
