"""One x86 read-head microstep, as a publishable program.

The decoder itself is ``src.compiler.x86_tensor_read_head`` -- this file adds
no decoding of its own, it only names an entry point and a subject so the
ordinary compiler/publisher route has a program to compile. The read head is
the repository's canonical *integral* program (20 int64 registers per lane,
every update predicated by a mask rather than a branch), which is what makes
it the natural first state machine to publish.
"""

from src.compiler.x86_tensor_read_head import (
    X86EncodingRow,
    X86ReadBatch,
    X86ReadHeadConfig,
    X86ReadHeadState,
    X86TensorReadHead,
)

TURING_PAGE = {
    "entrypoint": "read_head_step",
    "title": "x86 Read Head State Machine",
    "slug": "x86-read-head-state-machine",
    "feeds": {
        "octets": [[0x90, 0xC3], [0xC3, 0x00]],
        "valid_lengths": [2, 1],
        "base_addresses": [4096, 4352],
    },
}

# A deliberately tiny encoding table: NOP and RET. The point here is the
# state machine, not instruction coverage -- the table is data the head
# reads, so a larger one changes the buffer contents, not the program.
CONFIG = X86ReadHeadConfig.from_rows((
    X86EncodingRow(token=1, opcode_map=0, opcode=0x90),
    X86EncodingRow(token=2, opcode_map=0, opcode=0xC3, terminal=True),
))
HEAD = X86TensorReadHead(CONFIG)


def read_head_step(octets, valid_lengths, base_addresses):
    """Advance every lane by one decoder microstep and publish its phase."""

    batch = X86ReadBatch(
        octets=octets,
        valid_lengths=valid_lengths,
        base_addresses=base_addresses,
    )
    state = X86ReadHeadState.initial(batch)
    state = HEAD.transition(batch, state)
    return state.phase
