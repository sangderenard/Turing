import struct

from examples.meowdoku.fabric_udp import FabricUdpReceiver, HEADER, MAGIC, PAIR


def packet(frame, chunk, chunks, points, identity=42, state=0):
    return (HEADER.pack(MAGIC, 1, HEADER.size, frame, identity, chunk, chunks, len(points), state)
            + b"".join(PAIR.pack(node, *xyz) for node, xyz in points))


def test_complete_out_of_order_fabric_frame_only():
    receiver = FabricUdpReceiver()
    first = packet(7, 0, 2, [(0, (1.0, 2.0, 3.0))])
    second = packet(7, 1, 2, [(1, (4.0, 5.0, 6.0))])
    assert receiver.accept(second) is False
    assert receiver.latest(42) is None
    assert receiver.accept(first) is True
    frame = receiver.latest(42)
    assert frame["frame"] == 7
    assert frame["positions"] == {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}
    assert receiver.accept(first) is False


def test_bad_packet_cannot_publish():
    receiver = FabricUdpReceiver()
    bad = packet(1, 0, 1, [(256, (1.0, 2.0, 3.0))])
    assert receiver.accept(bad) is False
    nan = packet(1, 0, 1, [(0, (float("nan"), 2.0, 3.0))])
    assert receiver.accept(nan) is False
    assert receiver.latest(42) is None
