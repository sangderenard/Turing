from src.transmogrifier.bitbitbuffer.bitbitbuffer import BitBitBuffer


def test_get_by_pid_returns_absolute_gap():
    buffer = BitBitBuffer(mask_size=64)
    # Register PIDBuffer domain from 16 to 32 with stride 4
    buffer.register_pid_buffer(left=16, right=32, stride=4, label="A")
    pb = buffer.pid_buffers["A"]
    # Request a PID for absolute gap 20
    pid = pb.get_pids([20])[0]
    # Lookup should return the absolute gap 20
    assert buffer.get_by_pid("A", pid) == 20
