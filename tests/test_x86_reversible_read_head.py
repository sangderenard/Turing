import pytest

from src.common.tensors import AbstractTensor as AT
from src.compiler.x86_read_head_shader import build_x86_read_head_register_shader
from src.compiler.x86_tensor_read_head import (
    ReadHeadDirection,
    ReadPhase,
    X86EncodingRow,
    X86ReadBatch,
    X86ReadHeadConfig,
    X86ReadHeadState,
    X86ReversibleReadHead,
    X86TensorReadHead,
)


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
