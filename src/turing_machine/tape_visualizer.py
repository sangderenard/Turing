from __future__ import annotations
import shutil

class TapeVisualizer:
    """Render an ASCII bar of the tape with head position."""

    def __init__(self, machine: "TapeMachine", bar_width: int | None = None):
        self.machine = machine
        if bar_width is not None:
            self.bar_width = bar_width
        else:
            terminal_width = shutil.get_terminal_size().columns
            self.bar_width = max(20, terminal_width - 20)

    def _active_length(self) -> int:
        if not self.machine.tape_map:
            return len(self.machine.transport)
        max_reg = max(self.machine.data_registers.values(), default=0)
        max_pos = max(max_reg + self.machine.bit_width, self.machine.transport._cursor)
        return max_pos if max_pos > 0 else len(self.machine.transport)

    def _scale(self, bit_pos: int) -> int:
        tape_len = self._active_length()
        if tape_len <= 0:
            return 0
        scaled = int((bit_pos / tape_len) * self.bar_width)
        return max(0, min(self.bar_width - 1, scaled))

    def draw(self) -> None:
        if not self.machine.tape_map:
            return
        tape_map = self.machine.tape_map
        bar = ['#'] * self.bar_width
        bios_end = self._scale(tape_map.instr_start)
        instr_end = self._scale(tape_map.data_start)
        for i in range(bios_end):
            bar[i] = 'B'
        for i in range(bios_end, instr_end):
            bar[i] = 'I'
        head_pos = self._scale(self.machine.transport._cursor)
        bar[head_pos] = '@'
        display_str = f"Tape: [{''.join(bar)}]"
        print(display_str, end='\r')
