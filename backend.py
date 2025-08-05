import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

import src.hardware.analog_spec as analog_spec
sys.modules["analog_spec"] = analog_spec

from src.hardware.analog_spec import BiosHeader, InstructionWord, Opcode
from src.hardware.cassette_tape import CassetteTapeBackend
from src.turing_machine.tape_map import TapeMap
from src.turing_machine.tape_machine import TapeMachine


def _prepare_demo_tape() -> tuple[CassetteTapeBackend, tuple[int, int, int], int]:
    """Create a small tape with a single NAND instruction.

    Returns the cassette instance, the register addresses used, and the
    instruction count.  Registers ``R0`` and ``R1`` are preloaded with values
    ``1`` and ``0`` respectively so that executing the NAND stores ``1`` in
    ``R2``.
    """

    bios = BiosHeader(10.0, 50.0, 0.0, [], [], 0)
    instructions = [InstructionWord(Opcode.NAND, 0, 1, 2, 0)]
    tape_map = TapeMap(bios, instruction_frames=len(instructions))
    tape_map.bios.instr_start_addr = tape_map.instr_start

    cassette = CassetteTapeBackend(time_scale_factor=0.0, tape_length=1024)

    # Write BIOS frames
    for frame_idx, frame in enumerate(tape_map.encode_bios()):
        for lane, bit in enumerate(frame):
            cassette.write_bit(0, lane, tape_map.bios_start + frame_idx, bit)

    # Encode and write instruction frames using the TapeMachine layout
    def encode(instr: InstructionWord) -> list[int]:
        word = (
            (instr.opcode.value & 0xF) << 12
            | (instr.reg_a & 0x3) << 10
            | (instr.reg_b & 0x3) << 8
            | (instr.dest & 0x3) << 6
            | (instr.param & 0x3F)
        )
        return [(word >> (15 - i)) & 1 for i in range(16)]

    for frame_idx, instr in enumerate(instructions):
        frame = encode(instr)
        for lane, bit in enumerate(frame):
            cassette.write_bit(0, lane, tape_map.instr_start + frame_idx, bit)

    # Preload registers: R0=1, R1=0, R2=0
    r0 = tape_map.data_start
    r1 = tape_map.data_start + 1
    r2 = tape_map.data_start + 2
    cassette.write_bit(0, 0, r0, 1)
    cassette.write_bit(0, 0, r1, 0)
    cassette.write_bit(0, 0, r2, 0)

    return cassette, (r0, r1, r2), len(instructions)


if __name__ == "__main__":
    # Run pytest
    print("Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print("Tests failed.")
        sys.exit(result.returncode)

    print("Tests passed. Running tape simulator...")
    cassette, (r0, r1, r2), instr_count = _prepare_demo_tape()
    print(
        f"Register state before: R0={cassette.read_bit(0,0,r0)} "
        f"R1={cassette.read_bit(0,0,r1)} R2={cassette.read_bit(0,0,r2)}"
    )

    machine = TapeMachine(cassette, bit_width=1)
    machine.run(instr_count)

    print(
        f"Register state after: R0={cassette.read_bit(0,0,r0)} "
        f"R1={cassette.read_bit(0,0,r1)} R2={cassette.read_bit(0,0,r2)}"
    )
    cassette.close()
