# cassette_compat.py  —  keep for as long as older code needs it
from .cassette_tape import CassetteTapeBackend as _Core

class CassetteTapeBackend(_Core):
    """Legacy façade around the v2 physical simulator."""
    # alias old attr
    @property
    def tape_length(self) -> int:
        return self.total_bits

    # relative seek expected by GraphExecutor
    def move_head(self, delta: int) -> None:
        bit_pos = int(self._head_pos_inches * self.bits_per_inch) + delta
        self.move_head_to_bit(bit_pos)

    # no-op ‘instruction’ pulse so demos don’t break
    def execute_instruction(self) -> None:
        self._simulate_movement(0.0, 0.0, 1, "instr")
