from src.compiler.recursive_reduction import (
    TerminalTapeProgram,
    execute_terminal_tape_program,
)
from src.compiler.tape_compiler import TapeCompiler
from src.hardware.analog_spec import BiosHeader, InstructionWord, Opcode
from src.turing_machine.tape_map import TapeMap
from src.turing_machine.turing_provenance import ProvenanceGraph
from src.transmogrifier.ssa import Instr, SSAValue


def test_ssa_register_pressure_emits_and_executes_explicit_spill_abi():
    values = [SSAValue(index) for index in range(7)]
    instructions = [
        Instr("nand", [values[0], values[1]], values[4]),
        Instr("nand", [values[2], values[3]], values[5]),
        Instr("nand", [values[4], values[5]], values[6]),
    ]
    compiler = TapeCompiler(ProvenanceGraph(), bit_width=4)

    _map, encoded, _pcm = compiler.compile_ssa(instructions)

    assert compiler.spill_mode
    assert set(compiler.data_map) == set(range(7))
    assert not (set(compiler.memory_map) & set(range(7)))
    assert len(compiler.instruction_value_ids) == len(encoded)
    assert Opcode.LOAD in {instruction.opcode for instruction in encoded}
    assert Opcode.STORE in {instruction.opcode for instruction in encoded}
    assert all(
        instruction.param == compiler.data_map[value_id]
        for instruction, value_id in zip(encoded, compiler.instruction_value_ids)
        if instruction.opcode in {Opcode.LOAD, Opcode.STORE}
    )

    terminal = list(encoded)
    terminal.append(InstructionWord(
        Opcode.LOAD,
        reg_a=0,
        reg_b=0,
        dest=0,
        param=compiler.data_map[6],
    ))
    terminal.append(InstructionWord(
        Opcode.HALT,
        reg_a=0,
        reg_b=0,
        dest=0,
        param=0,
    ))
    bios = BiosHeader(10.0, 50.0, 1.0, [], [0], 0)
    tape_map = TapeMap(bios, instruction_frames=len(terminal))
    tape_map.bios.instr_start_addr = tape_map.instr_start
    frames = TapeCompiler.binarize_instructions(terminal)
    inputs = {0: 0b1100, 1: 0b1010, 2: 0b1111, 3: 0b0011}
    program = TerminalTapeProgram(
        tape_map=tape_map,
        instructions=tuple(terminal),
        instruction_frames=tuple(tuple(frame) for frame in frames),
        instruction_sources=tuple(
            compiler.instruction_value_ids + [6, None]
        ),
        initial_register_values={},
        initial_spill_values={
            compiler.data_map[value_id]: value
            for value_id, value in inputs.items()
        },
        spill_slots=dict(compiler.data_map),
        node_registers={6: 0},
        output_registers={6: 0},
        output_spill_slots={},
        bit_width=4,
        storage_mode="spilled",
    )

    witness = execute_terminal_tape_program(program)

    first = (~(inputs[0] & inputs[1])) & 0b1111
    second = (~(inputs[2] & inputs[3])) & 0b1111
    expected = (~(first & second)) & 0b1111
    assert witness.halted
    assert witness.outputs == {6: expected}
