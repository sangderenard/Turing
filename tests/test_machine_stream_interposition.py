from types import MappingProxyType

from src.compiler.amd64_machine_semantics import (
    condition_holds, default_effect_handlers, indirect_target,
)
from src.compiler.machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionState,
)
from src.compiler.machine_reference_vocabulary import X86ReferenceDecoder
from src.compiler.live_ssa_execution import LiveSSAExecutionSession
from src.compiler.machine_dialect_ssa import decoded_function_to_machine_ssa
from src.compiler.machine_stream_interposition import (
    BidirectionalSSAWriteHead, MachineInstructionStreamInterposer, MachineStreamRoute,
    MachineStreamBlockDispatcher, RecompiledMachineStream,
    SSAExternalCodeReference, SSAWriteHeadRequest,
)
from src.compiler.live_ssa_execution import SSAEditTransaction, address_linked_ssa_lines
from src.compiler.machine_code_lifting import raise_binary_region_to_ssa
from src.compiler.machine_reference_vocabulary import (
    ImmediateOperand, RegisterOperand, X86InstructionToken, X86Register,
)
from src.compiler.x86_tensor_read_head import (
    X86AllocatedInstruction, X86ReadHeadProfile, X86TensorReadHead,
    controlled_x86_64_read_head_profile,
)
from src.transmogrifier.ssa import IRModule
from src.compiler.machine_execution import ReversibleMachineExecutor
from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_state_buffer import MachineRunDirection


class _Image:
    image_base = 0x1000
    entrypoint_rva = 0
    encoded = None
    sections = ()


class _Report:
    def __init__(self, instructions):
        self.instructions = tuple(instructions)


class _Record:
    def __init__(self, instructions):
        self.report = _Report(instructions)


class _Program:
    image = _Image()

    def __init__(self, instructions):
        self.functions = (_Record(instructions),)


def _executor(original):
    decoder = X86ReferenceDecoder()
    decoded = []
    cursor = 0
    while cursor < len(original):
        instruction, cursor = decoder.decode_one(
            memoryview(original), cursor, base_address=0x1000,
        )
        decoded.append(instruction)
    return MachineExecutionOrchestrator(
        _Program(decoded), effect_handlers=default_effect_handlers(),
        predicate_handler=condition_holds,
        indirect_target_handler=indirect_target,
    )


def _state(original):
    return MachineExecutionState(
        pc=0x1000,
        memory=MappingProxyType({
            0x1000 + index: value for index, value in enumerate(original)
        }),
    )


def test_recompiled_stream_uses_read_head_without_changing_original_bytes():
    original = b"\x48\x83\xc0\x01\xc3"  # add rax, 1; ret
    replacement = b"\x48\x83\xc0\x30"   # add rax, 48
    stream = RecompiledMachineStream(
        "edited_ssa", {0x7000: replacement}, {0x1000: 0x7000},
        {0x7004: 0x1004}, {0x7000: ("subject:entry:2",)},
        {0x7000: "proof-witness"},
    )
    interposer = MachineInstructionStreamInterposer(_executor(original), stream)
    state = _state(original)

    result = interposer.step(state)
    event = interposer.last_stream_event

    assert result.state.pc == 0x1004
    assert result.state.registers[0] == 48
    assert bytes(state.memory[0x1000 + index] for index in range(4)) == original[:4]
    assert event is not None
    assert event.encoded == replacement
    assert event.route is MachineStreamRoute.SSA_RECOMPILE
    assert event.source_encoded == original[:4]
    assert event.instruction_address == 0x7000
    assert event.redirected_to == 0x1004
    assert event.ssa_line_ids == ("subject:entry:2",)
    assert event.witness == "proof-witness"
    assert event.read_head_microsteps > 0
    assert event.source_read_head_microsteps > 0


def test_trigger_reads_original_before_same_address_write_stream_supersedes_it():
    original = b"\x48\x83\xc0\x01\xc3"
    replacement = b"\x48\x83\xc0\x30"
    stream = RecompiledMachineStream(
        "same_address", {0x1000: replacement}, {0x1000: 0x1000},
        {0x1004: 0x1004},
    )
    pipeline = MachineInstructionStreamInterposer(_executor(original), stream)

    result = pipeline.step(_state(original))
    event = pipeline.last_stream_event

    assert result.state.registers[0] == 48
    assert event is not None
    assert event.source_encoded == original[:4]
    assert event.encoded == replacement


def test_large_replacement_stream_executes_multiple_instructions_then_resumes():
    original = b"\x48\x83\xc0\x01\xc3"
    stream = RecompiledMachineStream(
        "large_edited_ssa",
        {
            0x7000: b"\x48\x83\xc0\x02",  # add rax, 2
            0x7004: b"\x48\x83\xc0\x03",  # add rax, 3
        },
        {0x1000: 0x7000}, {0x7008: 0x1004},
        {
            0x7000: ("large:entry:4", "large:entry:5"),
            0x7004: ("large:entry:6", "large:entry:7"),
        },
    )
    interposer = MachineInstructionStreamInterposer(_executor(original), stream)

    first = interposer.step(_state(original))
    first_event = interposer.last_stream_event
    second = interposer.step(first.state)
    second_event = interposer.last_stream_event

    assert first.state.pc == 0x7004
    assert first.state.registers[0] == 2
    assert second.state.pc == 0x1004
    assert second.state.registers[0] == 5
    assert first_event is not None and second_event is not None
    assert first_event.instruction_address == 0x7000
    assert second_event.instruction_address == 0x7004
    assert second_event.redirected_to == 0x1004


def test_read_head_precedes_direct_pass_through_to_executor():
    original = b"\x48\x83\xc0\x01\xc3"
    decoder = X86ReferenceDecoder()
    decoded = decoder.decode_report(
        original, base_address=0x1000, stop_at_return=False,
        allow_trailing_after_terminal=True,
    ).instructions
    source_module = IRModule({
        "source": decoded_function_to_machine_ssa("source", decoded),
    })
    pipeline = MachineInstructionStreamInterposer(
        _executor(original), source_ssa=source_module,
    )

    result = pipeline.step(_state(original))
    event = pipeline.last_stream_event

    assert result.state.registers[0] == 1
    assert event is not None
    assert event.route is MachineStreamRoute.PASS_THROUGH
    assert event.source_encoded == original[:4]
    assert event.encoded == original[:4]
    assert event.read_head_microsteps > 0
    assert event.ssa_line_ids
    assert all(item.startswith("source:") for item in event.ssa_line_ids)


def test_ssa_external_code_reference_supersedes_executor_head_not_runtime():
    original = b"\x48\x83\xc0\x01\xc3"
    lifting = raise_binary_region_to_ssa(
        original, maximum_file_size=len(original), size=len(original),
        base_address=0x8000, name="referenced_large_ssa",
        full_vocabulary_report=True, cfg_decode=True,
    )
    module = IRModule({"referenced_large_ssa": lifting.function})
    constant = next(
        line for line in address_linked_ssa_lines(module)
        if line.machine_address == 0x8000 and line.operation == "Const"
    )
    SSAEditTransaction(module).replace_constant(constant.line_id, 7)
    request = SSAWriteHeadRequest(
        module, "referenced_large_ssa", 0x1000, 0x8000,
        {0x8004: 0x1004},
        {0x8000: {
            "register-or-memory-destination", "signed-immediate-8",
            "width-64", "modulo-2^64", "all-add-flags-exact",
        }},
        allocated_instructions_by_address={0x8000: X86AllocatedInstruction(
            int(X86InstructionToken.ADD_R64_IMM8),
            (
                RegisterOperand(X86Register.RAX, 64),
                ImmediateOperand(7, 8, True),
            ),
            0x8000,
        )},
        allow_new_selection=True,
    )
    reference = SSAExternalCodeReference(41, "alternate_palette", request)
    pipeline = MachineInstructionStreamInterposer(
        _executor(original), external_code_references=(reference,),
    )

    result = pipeline.step(_state(original))
    event = pipeline.last_stream_event

    assert result.state.registers[0] == 7
    assert event is not None
    assert event.route is MachineStreamRoute.EXTERNAL_SSA_REFERENCE
    assert event.external_reference_id == 41
    assert event.instruction_address == 0x8000


def test_live_viewport_highlights_executed_stream_ssa_not_trigger_address():
    original = b"\x48\x83\xc0\x01\xc3"
    decoder = X86ReferenceDecoder()
    replacement, _end = decoder.decode_one(
        memoryview(b"\x48\x83\xc0\x09"), 0, base_address=0x9000,
    )
    module = IRModule({
        "replacement": decoded_function_to_machine_ssa(
            "replacement", (replacement,),
        ),
    })
    line_ids = tuple(
        f"replacement:{block_name}:{ordinal}"
        for block_name, block in module.functions["replacement"].blocks.items()
        for ordinal, instruction in enumerate(block.instrs)
        if instruction.attributes.get("machine_address") == 0x9000
    )
    stream = RecompiledMachineStream(
        "replacement", {0x9000: replacement.encoded}, {0x1000: 0x9000},
        {0x9004: 0x1004}, {0x9000: line_ids},
    )
    pipeline = MachineInstructionStreamInterposer(_executor(original), stream)
    session = LiveSSAExecutionSession(module, pipeline, _state(original))

    event = session.step()

    assert event.source_machine_address == 0x1000
    assert event.machine_address == 0x9000
    assert event.stream_route == "SSA_RECOMPILE"
    assert event.stream_name == "replacement"
    assert event.source_encoded == original[:4]
    assert event.highlighted_line_ids == line_ids


def test_write_head_emits_one_repository_ssa_block_without_whole_module_write():
    lifting = raise_binary_region_to_ssa(
        b"\x74\x03\x83\xc0\x01\xc3", maximum_file_size=6, size=6,
        base_address=0xA000, name="scoped", full_vocabulary_report=True,
        cfg_decode=True,
    )
    module = IRModule({"scoped": lifting.function})
    block_name, addresses = next(
        (name, addresses)
        for name, addresses in lifting.function.metadata["machine_block_addresses"]
        if addresses
    )
    request = SSAWriteHeadRequest(
        module, "scoped-fragment", 0x1000,
        stream_entry_address=int(addresses[0]),
    )

    write_head = BidirectionalSSAWriteHead()
    stream = write_head.compile_block(
        request, "scoped", block_name,
    )
    cached = write_head.compile_block(request, "scoped", block_name)

    assert tuple(stream.instructions) == tuple(addresses)
    assert cached is stream
    assert write_head.cache_statistics == {
        "fragments": 1, "hits": 1, "misses": 1,
    }


def test_write_head_emits_existing_region_function_as_all_of_its_blocks():
    lifting = raise_binary_region_to_ssa(
        b"\x74\x03\x83\xc0\x01\xc3", maximum_file_size=6, size=6,
        base_address=0xB000, name="planned_region_7",
        full_vocabulary_report=True, cfg_decode=True,
    )
    module = IRModule({"planned_region_7": lifting.function})
    expected = tuple(dict.fromkeys(
        address
        for _block_name, addresses
        in lifting.function.metadata["machine_block_addresses"]
        for address in addresses
    ))
    request = SSAWriteHeadRequest(
        module, "region-fragment", 0x1000,
        stream_entry_address=min(expected),
    )

    stream = BidirectionalSSAWriteHead().compile_function(
        request, "planned_region_7",
    )

    assert tuple(stream.instructions) == expected


def test_write_head_cache_does_not_consume_one_shot_proof_facts():
    lifting = raise_binary_region_to_ssa(
        b"\x83\xc0\x01\xc3", maximum_file_size=4, size=4,
        base_address=0xC000, name="proof_facts",
        full_vocabulary_report=True, cfg_decode=True,
    )
    module = IRModule({"proof_facts": lifting.function})
    request = SSAWriteHeadRequest(
        module, "proof-facts", 0x1000,
        proven_facts_by_address={
            0xC000: (
                fact for fact in (
                    "register-or-memory-destination", "signed-immediate-8",
                    "width-64", "modulo-2^64", "all-add-flags-exact",
                )
            ),
        },
    )

    stream = BidirectionalSSAWriteHead().compile(request)

    assert stream.instructions[0xC000] == b"\x83\xc0\x01"


def test_stream_dispatcher_commits_each_injected_instruction_reversibly():
    original = b"\x48\x83\xc0\x01\xc3"
    stream = RecompiledMachineStream(
        "runner-stream",
        {
            0xD000: b"\x48\x83\xc0\x02",
            0xD004: b"\x48\x83\xc0\x03",
        },
        {0x1000: 0xD000}, {0xD008: 0x1004},
    )
    executor = _executor(original)
    initial = _state(original)
    core = ReversibleMachineExecutor.create(executor, initial)
    dispatcher = MachineStreamBlockDispatcher(
        MachineInstructionStreamInterposer(executor, stream),
    )

    results = dispatcher.execute(core, 8)

    assert results is not None and len(results) == 2
    assert core.state.pc == 0x1004
    assert core.state.registers[0] == 5
    assert core.position == 2
    assert core.step_backward().registers[0] == 2
    assert core.step_backward() == initial
    assert dispatcher.statistics["stream_committed_instructions"] == 2


def test_stream_dispatcher_delegates_unclaimed_guest_code_to_existing_backend():
    original = b"\x48\x83\xc0\x01\xc3"
    executor = _executor(original)
    stream = RecompiledMachineStream(
        "inactive", {0xD000: b"\xc3"}, {0x2000: 0xD000}, {},
    )
    delegated = []

    class ExistingDispatcher:
        statistics = {"executions": 7}

        def execute(self, core, maximum_instructions, *, transition_observer=None):
            delegated.append((core.state.pc, maximum_instructions, transition_observer))
            return (core.step_forward(),)

    core = ReversibleMachineExecutor.create(executor, _state(original))
    dispatcher = MachineStreamBlockDispatcher(
        MachineInstructionStreamInterposer(executor, stream),
        fallback_dispatcher=ExistingDispatcher(),
    )

    results = dispatcher.execute(core, 4)

    assert results is not None and results[0].state.registers[0] == 1
    assert delegated and delegated[0][:2] == (0x1000, 4)
    assert dispatcher.statistics["fallback_executions"] == 7


def test_binary_machine_installs_stream_ahead_of_existing_runner_backend():
    original = b"\x48\x83\xc0\x01\xc3"  # add rax, 1; ret
    executor = _executor(original)
    program = BinaryMachineProgram.from_program(
        executor.program, effect_handlers=default_effect_handlers(),
    )
    stream = RecompiledMachineStream(
        "live-edit", {0xD000: b"\x48\x83\xc0\x2a"},
        {0x1000: 0xD000}, {0xD004: 0x1004},
    )
    interposer = MachineInstructionStreamInterposer(
        program.machine.cores[0].executor, stream,
    )
    prior = program.runner.compiled_dispatcher
    try:
        program.install_stream_interposer(interposer)
        program.set_direction(MachineRunDirection.FORWARD)

        assert program.runner.tick(2) == 2
        assert program.machine.cores[0].state.registers[0] == 42
        assert program.runner._last_results[0].status.name == "HALTED"

        program.remove_stream_interposer()
        assert program.runner.compiled_dispatcher is prior
    finally:
        program.close()


def test_complete_reference_head_admits_token_outside_tensor_accelerator():
    original = b"\xb8\x09\x00\x00\x00\xc3"  # mov eax, 9; ret
    executor = _executor(original)
    # Production acceleration now covers the complete authoritative table.
    # Install a deliberately reduced deployment profile to prove accelerator
    # omission still cannot narrow reference-head decompilation.
    complete_profile = controlled_x86_64_read_head_profile()
    reduced_profile = X86ReadHeadProfile(
        "test-reduced-accelerator",
        tuple(
            row for row in complete_profile.rows
            if row.token != int(X86InstructionToken.NOP)
        ),
        escape_maps=complete_profile.escape_maps,
    )
    stream = RecompiledMachineStream(
        "reference-framed", {0xD000: b"\x90"},
        {0x1000: 0xD000}, {0xD001: 0x1005},
    )
    pipeline = MachineInstructionStreamInterposer(executor, stream)
    pipeline._read_head = X86TensorReadHead.from_profile(reduced_profile)

    result = pipeline.step(_state(original))
    event = pipeline.last_stream_event

    assert result.state.pc == 0x1005
    assert event is not None
    assert event.source_read_head == "tensor-verified"
    assert event.read_head == "reference"
    assert event.source_read_head_microsteps > 0
    assert event.read_head_microsteps == 0
