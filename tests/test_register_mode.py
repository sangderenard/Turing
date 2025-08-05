import pytest

from src.hardware.cassette_tape import CassetteTapeBackend
from src.turing_machine.tape_transport import TapeTransport


def test_register_transport_requires_operator_sequence():
    tape = CassetteTapeBackend(tape_length_inches=0.02, time_scale_factor=0.0)
    transport = TapeTransport(tape, register_mode=True)
    try:
        with pytest.raises(PermissionError):
            _ = transport[0]
        transport.queue_operators([0x1])
        _ = transport[0]
        with pytest.raises(PermissionError):
            _ = transport[1]
    finally:
        tape.close()
