import pytest

from src.common.tensors import AbstractTensor as AT
from src.compiler.x86_read_head_shader import build_x86_read_head_register_shader
from src.compiler.x86_tensor_read_head import (
    ReadHeadDirection,
    ReadHeadExecutionMode,
    ReadFailure,
    ReadPhase,
    ReadStatus,
    X86EncodingRow,
    X86EncodingFields,
    X86AllocatedInstruction,
    X86ReadBatch,
    X86ReadHeadConfig,
    X86ReadHeadCodeSetBank,
    X86ReadHeadState,
    X86ReversibleReadHead,
    X86TensorReadHead,
    controlled_x86_64_read_head_profile,
)
from src.compiler.machine_reference_vocabulary import (
    X86InstructionToken, X86ReferenceDecoder, X86_64_REFERENCE_VOCABULARY,
)


def test_locked_add_bidirectional_row_requires_memory_modrm():
    profile = controlled_x86_64_read_head_profile()
    token = int(X86InstructionToken.LOCK_ADD_RM8_R8)
    row = next(item for item in profile.rows if item.token == token)
    assert row.modrm_memory_only is True
    assert profile.encode(token, X86EncodingFields(modrm=0x08)) == b"\xf0\x00\x08"
    with pytest.raises(ValueError, match="memory ModRM"):
        profile.encode(token, X86EncodingFields(modrm=0xC8))
    with pytest.raises(Exception, match="memory destination"):
        X86ReferenceDecoder().decode_one(memoryview(b"\xf0\x00\xc8"), 0)

    head = X86TensorReadHead.from_profile(profile)
    encoded = b"\xf0\x00\xc8"
    result = head.run(X86ReadBatch(
        octets=AT.get_tensor([list(encoded)], dtype="int64"),
        valid_lengths=AT.get_tensor([len(encoded)], dtype="int64"),
        base_addresses=AT.get_tensor([0x1000], dtype="int64"),
    ))
    assert result.final_state.status.item() == int(ReadStatus.FAILED)
    assert result.final_state.failure.item() == int(ReadFailure.INVALID_MODRM)


def test_bidirectional_profile_is_derived_from_authoritative_specs():
    specs = {
        int(spec.token): spec for spec in X86_64_REFERENCE_VOCABULARY
        if spec.reversible_operand_form is not None
    }
    profile = controlled_x86_64_read_head_profile()

    assert {row.token for row in profile.rows} == set(specs)
    for row in profile.rows:
        spec = specs[row.token]
        assert row.operand_form.name == spec.reversible_operand_form
        assert row.immediate_bytes == spec.reversible_immediate_bytes
        assert row.immediate_signed == spec.reversible_immediate_signed
        assert row.immediate_relative == spec.reversible_immediate_relative
        assert row.terminal == spec.reversible_terminal


def test_every_authoritative_token_has_reference_and_tensor_roundtrip():
    profile = controlled_x86_64_read_head_profile()
    head = X86TensorReadHead.from_profile(profile)
    reference = X86ReferenceDecoder()
    encodings = []
    expected_tokens = []
    for row in profile.rows:
        # Register-direct is a valid canonical ModRM for nearly every row;
        # LEA is the one grammar that requires a memory effective address.
        modrm = None
        if row.has_modrm:
            extension = 0 if row.modrm_extension is None else row.modrm_extension
            mod = 0 if row.modrm_memory_only or row.token in {
                int(X86InstructionToken.LEA_R32_M),
                int(X86InstructionToken.LEA_R64_M),
            } else 3
            modrm = (mod << 6) | (extension << 3)
        encoded = profile.encode(row.token, X86EncodingFields(
            modrm=modrm,
            immediate=0 if row.immediate_bytes else None,
        ))
        decoded, end = reference.decode_one(memoryview(encoded), 0)
        assert end == len(encoded), X86InstructionToken(row.token).name
        assert int(decoded.token) == row.token
        assert head.rewrite_instruction(row.token, encoded) == encoded
        encodings.append(encoded)
        expected_tokens.append(row.token)

    capacity = max(map(len, encodings))
    batch = X86ReadBatch(
        octets=AT.get_tensor([
            list(encoded) + [0] * (capacity - len(encoded))
            for encoded in encodings
        ], dtype="int64"),
        valid_lengths=AT.get_tensor(list(map(len, encodings)), dtype="int64"),
        base_addresses=AT.get_tensor(
            [0x1000 + index * 0x20 for index in range(len(encodings))],
            dtype="int64",
        ),
    )
    result = head.run(batch, mode=ReadHeadExecutionMode.TRACE)
    observed = [None] * len(encodings)
    for state in result.emission_states:
        for lane, (status, token) in enumerate(zip(
            state.status.tolist(), state.token.tolist(),
        )):
            if observed[lane] is None and status in {
                int(ReadStatus.EMITTED), int(ReadStatus.HALTED),
            }:
                observed[lane] = token
    assert observed == expected_tokens


def test_namespaced_code_sets_decode_and_write_without_global_token_identity():
    source = controlled_x86_64_read_head_profile()
    first = source.namespace("first", token_base=1000)
    second = source.namespace("second", token_base=2000)
    source_ret = int(X86InstructionToken.RET_NEAR)
    first_ret = next(
        token for token, original in first.source_tokens.items()
        if original == source_ret
    )
    second_ret = next(
        token for token, original in second.source_tokens.items()
        if original == source_ret
    )

    assert first_ret != second_ret
    assert first.token_name(first_ret) == "first::RET_NEAR"
    assert second.token_name(second_ret) == "second::RET_NEAR"
    assert first.encode(first_ret) == second.encode(second_ret) == b"\xc3"
    assert X86TensorReadHead.from_profile(first).rewrite_instruction(
        first_ret, b"\xc3"
    ) == b"\xc3"
    assert X86TensorReadHead.from_profile(second).rewrite_instruction(
        second_ret, b"\xc3"
    ) == b"\xc3"
    with pytest.raises(ValueError, match="exactly one encoding row"):
        first.encode(second_ret)

    def decoded_token(profile):
        head = X86TensorReadHead.from_profile(profile)
        result = head.run(X86ReadBatch(
            octets=AT.get_tensor([[0xC3]], dtype="int64"),
            valid_lengths=AT.get_tensor([1], dtype="int64"),
            base_addresses=AT.get_tensor([0x1000], dtype="int64"),
        ), mode=ReadHeadExecutionMode.TRACE)
        return result.emission_states[0].token.item()

    assert decoded_token(first) == first_ret
    assert decoded_token(second) == second_ret

    bank = X86ReadHeadCodeSetBank({"first": first, "second": second})
    assert bank.token_owner(first_ret) == "first"
    assert bank.token_owner(second_ret) == "second"
    assert bank.rewrite_instruction("first", first_ret, b"\xc3") == b"\xc3"
    assert bank.rewrite_instruction("second", second_ret, b"\xc3") == b"\xc3"
    with pytest.raises(ValueError, match="exactly one encoding row"):
        bank.encode("first", second_ret)
    batches = {
        name: X86ReadBatch(
            octets=AT.get_tensor([[0xC3]], dtype="int64"),
            valid_lengths=AT.get_tensor([1], dtype="int64"),
            base_addresses=AT.get_tensor([0x1000], dtype="int64"),
        )
        for name in ("first", "second")
    }
    simultaneous = bank.run(batches, mode=ReadHeadExecutionMode.TRACE)
    assert simultaneous["first"].emission_states[0].token.item() == first_ret
    assert simultaneous["second"].emission_states[0].token.item() == second_ret


def test_code_set_bank_rejects_unrenamed_token_collisions():
    source = controlled_x86_64_read_head_profile()
    with pytest.raises(ValueError, match="namespace them first"):
        X86ReadHeadCodeSetBank({"first": source, "second": source})


def _runtime(*lanes):
    capacity = max(len(lane) for lane in lanes)
    padded = [list(lane) + [0] * (capacity - len(lane)) for lane in lanes]
    batch = X86ReadBatch(
        octets=AT.get_tensor(padded, dtype="int64"),
        valid_lengths=AT.get_tensor([len(lane) for lane in lanes], dtype="int64"),
        base_addresses=AT.get_tensor(
            [0x1000 + index * 0x100 for index in range(len(lanes))],
            dtype="int64",
        ),
    )
    config = X86ReadHeadConfig.from_rows((
        X86EncodingRow(token=1, opcode_map=0, opcode=0x90),
        X86EncodingRow(token=2, opcode_map=0, opcode=0xC3, terminal=True),
    ))
    return X86ReversibleReadHead.create(X86TensorReadHead(config), batch)


def test_virtual_cores_advance_concurrently_and_publish_every_register():
    runtime = _runtime((0x90, 0xC3), (0xC3,))

    runtime.transition()
    state = runtime.transition()

    assert runtime.core_count == 2
    assert state.phase.tolist() == [int(ReadPhase.EMIT), int(ReadPhase.EMIT)]
    assert tuple(runtime.register_tensor().shape) == (
        2, len(state.REGISTER_NAMES),
    )
    contents = runtime.register_contents()
    assert tuple(contents[0]) == state.REGISTER_NAMES
    assert contents[0]["cursor"] == 1
    assert contents[1]["token"] == 2


def test_reverse_restores_exact_partial_decode_and_forward_can_branch():
    runtime = _runtime((0x90, 0xC3),)
    initial = runtime.register_contents()
    runtime.transition()
    after_prefix = runtime.register_contents()
    runtime.transition()
    future_length = runtime.history_length

    assert runtime.transition(ReadHeadDirection.BACKWARD).register_contents() == after_prefix
    assert runtime.transition(ReadHeadDirection.BACKWARD).register_contents() == initial
    with pytest.raises(IndexError, match="beginning"):
        runtime.transition(ReadHeadDirection.BACKWARD)

    runtime.transition()
    assert runtime.history_length < future_length + 1
    assert runtime.history_position == 1


def test_fork_has_independent_reversible_future_and_acknowledgement():
    runtime = _runtime((0x90, 0xC3),)
    runtime.transition()
    branch = runtime.fork()

    branch.transition()
    branch.transition()  # EMIT -> status update
    branch.acknowledge()

    assert branch.history_position > runtime.history_position
    assert runtime.state.phase.tolist() == [int(ReadPhase.OPCODE)]
    assert branch.state.phase.tolist() == [int(ReadPhase.PREFIX)]


def test_register_shader_matches_packed_state_abi():
    artifact = build_x86_read_head_register_shader(workgroup_size=32)

    assert artifact.register_names == X86ReadHeadState.REGISTER_NAMES
    assert artifact.register_count == 20
    assert "@compute @workgroup_size(32)" in artifact.source
    assert "core_count * display.register_count" in artifact.source
    assert "array<i32>" in artifact.source


@pytest.mark.parametrize("encoded", (
    b"\x0f\xaf\xc1",                 # imul eax, ecx
    b"\x8d\x44\x8a\x10",             # lea eax, [rdx+rcx*4+16]
    b"\xc3",                           # ret
    b"\x48\x83\xe8\x01",             # sub rax, 1
    b"\xe8\x04\x00\x00\x00",         # call +4
    b"\x48\x83\xc0\x7f",             # add rax, 127
    b"\xe9\xfc\xff\xff\xff",         # jmp -4
    b"\x4c\x89\xc0",                 # mov rax, r8
    b"\x41\x50",                      # push r8
    b"\x48\x8b\x44\x24\x08",         # mov rax, [rsp+8]
    b"\x48\x83\xe0\xff",             # and rax, -1
    b"\x49\xb8\x08\x07\x06\x05\x04\x03\x02\x01",  # mov r8, imm64
    b"\x48\x3b\x45\xf8",             # cmp rax, [rbp-8]
    b"\x0f\x85\x04\x00\x00\x00",     # jne +4
    b"\x66\x0f\xd4\xc1",              # paddq xmm0, xmm1
    b"\x66\x0f\xfb\xc1",              # psubq xmm0, xmm1
    b"\xf3\x48\xa5",                  # rep movsq
    b"\xf0\x0f\xc1\x08",              # lock xadd [rax], ecx
    b"\x0f\xba\xf8\x03",              # btc eax, 3
    b"\x03\xc1",                       # add eax, ecx
    b"\x48\x03\xc1",                  # add rax, rcx
    b"\x66\x03\xc1",                  # add ax, cx
    b"\x66\x0f\x38\x29\xc1",          # pcmpeqq xmm0, xmm1
    b"\xf2\x0f\x58\xc1",               # addsd xmm0, xmm1
    b"\x75\xfe",                       # jne -2
    b"\x0f\x84\x04\x00\x00\x00",     # je +4
    b"\x0f\x94\xc0",                   # sete al
    b"\x48\xf7\xd0",                  # not rax
    b"\xff\xd0",                       # call rax
    b"\xff\xe0",                       # jmp rax
    b"\x48\xf7\xf1",                  # div rcx
    b"\xff\xc0",                       # inc eax
    b"\xf7\xd8",                       # neg eax
))
def test_bidirectional_head_writes_exact_controlled_instruction(encoded):
    instruction, end = X86ReferenceDecoder().decode_one(memoryview(encoded), 0)
    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())

    assert end == len(encoded)
    fields = head.encoding_fields(int(instruction.token), instruction.encoded)
    assert head.write_instruction(int(instruction.token), fields) == encoded
    assert head.rewrite_instruction(int(instruction.token), encoded) == encoded


def test_required_legacy_prefix_is_both_read_and_write_policy():
    from src.compiler.machine_reference_vocabulary import X86InstructionToken

    profile = controlled_x86_64_read_head_profile()
    head = X86TensorReadHead.from_profile(profile)

    token = int(X86InstructionToken.REP_MOVSQ)
    assert head.write_instruction(token) == b"\xf3\x48\xa5"
    with pytest.raises(ValueError, match="required legacy"):
        head.write_instruction(token, fields=head.encoding_fields(
            token, b"\xf3\x48\xa5",
        ).__class__(rex=0x48, legacy_prefixes=()))


def test_same_bidirectional_head_reads_the_reverse_compilation_forms():
    from src.compiler.machine_reference_vocabulary import X86InstructionToken

    encoded = (
        b"\x66\x0f\xd4\xc1"
        b"\x66\x0f\xfb\xc1"
        b"\xf3\x48\xa5"
        b"\xf0\x0f\xc1\x08"
        b"\x0f\xba\xf8\x03"
    )
    batch = X86ReadBatch(
        octets=AT.get_tensor([list(encoded)], dtype="int64"),
        valid_lengths=AT.get_tensor([len(encoded)], dtype="int64"),
        base_addresses=AT.get_tensor([0x1000], dtype="int64"),
    )
    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())
    result = head.run(batch, mode=ReadHeadExecutionMode.TRACE)

    assert tuple(state.token.item() for state in result.emission_states) == (
        int(X86InstructionToken.PADDQ_XMM_XMMM128),
        int(X86InstructionToken.PSUBQ_XMM_XMMM128),
        int(X86InstructionToken.REP_MOVSQ),
        int(X86InstructionToken.XADD_RM32_R32),
        int(X86InstructionToken.BTC_RM32_IMM8),
    )


def test_mov_r32_imm32_round_trips_through_shared_tensor_head():
    encoded = b"\xb8\x09\x00\x00\x00"
    batch = X86ReadBatch(
        octets=AT.get_tensor([list(encoded)], dtype="int64"),
        valid_lengths=AT.get_tensor([len(encoded)], dtype="int64"),
        base_addresses=AT.get_tensor([0x1000], dtype="int64"),
    )
    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())
    result = head.run(batch, mode=ReadHeadExecutionMode.TRACE)
    token = int(X86InstructionToken.MOV_R32_IMM32)

    assert tuple(state.token.item() for state in result.emission_states) == (token,)
    fields = head.encoding_fields(token, encoded)
    assert fields.opcode_low_bits == 0
    assert fields.immediate == 9
    assert head.write_instruction(token, fields) == encoded


def test_tensor_read_side_rejects_missing_mandatory_prefix():
    encoded = b"\x48\xa5"
    batch = X86ReadBatch(
        octets=AT.get_tensor([list(encoded)], dtype="int64"),
        valid_lengths=AT.get_tensor([len(encoded)], dtype="int64"),
        base_addresses=AT.get_tensor([0x1000], dtype="int64"),
    )
    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())
    result = head.run(batch, mode=ReadHeadExecutionMode.TRACE)

    assert result.final_state.failure.item() == int(ReadFailure.REQUIRED_PREFIX_MISSING)


@pytest.mark.parametrize(("token_name", "operands", "address", "encoded"), (
    ("MOV_R64_RM64", ("r8", ("rbx", "r12", 4, 16)), None, b"\x4e\x8b\x44\xa3\x10"),
    ("XADD_RM32_R32", (("rbx", None, 1, 8), "r9d"), None, b"\xf0\x44\x0f\xc1\x4b\x08"),
    ("BTC_RM32_IMM8", ("eax", 3), None, b"\x0f\xba\xf8\x03"),
    ("MOV_R64_IMM64", ("r8", 0x0102030405060708), None, b"\x49\xb8\x08\x07\x06\x05\x04\x03\x02\x01"),
    ("MOV_R32_IMM32", ("r9d", 0x01020304), None, b"\x41\xb9\x04\x03\x02\x01"),
    ("JNE_REL32", (0x1020,), 0x1000, b"\x0f\x85\x1a\x00\x00\x00"),
))
def test_allocated_operands_are_written_by_head(token_name, operands, address, encoded):
    from src.compiler.machine_reference_vocabulary import (
        EffectiveAddressOperand, ImmediateOperand, RegisterOperand,
        RelativeAddressOperand, X86InstructionToken, X86Register,
    )

    def register(name):
        aliases = {"eax": "RAX", "r9d": "R9"}
        return RegisterOperand(X86Register[aliases.get(name, name.upper())], 32 if name.endswith("d") or name == "eax" else 64)

    converted = []
    for item in operands:
        if isinstance(item, str):
            converted.append(register(item))
        elif isinstance(item, tuple):
            base, index, scale, displacement = item
            converted.append(EffectiveAddressOperand(
                register(base).register if base else None,
                register(index).register if index else None,
                scale, displacement,
            ))
        elif token_name in {"JNE_REL32"}:
            converted.append(RelativeAddressOperand(0, 32, item))
        else:
            converted.append(ImmediateOperand(
                item,
                8 if token_name == "BTC_RM32_IMM8" else (
                    32 if token_name == "MOV_R32_IMM32" else 64
                ),
                token_name == "BTC_RM32_IMM8",
            ))
    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())
    token = int(X86InstructionToken[token_name])

    assert head.write_allocated(X86AllocatedInstruction(
        token, tuple(converted), address,
    )) == encoded
