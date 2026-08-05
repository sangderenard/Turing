import struct
from threading import get_ident, Thread
from types import SimpleNamespace
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.compiler.machine_execution import MachineExecutionState
from src.compiler.machine_snapshot_host import (
    LiveMachineSnapshotController,
    MachineSnapshotMailbox,
    MachineTerminalInputQueue,
    build_machine_snapshot_server,
)
from src.compiler.machine_state_buffer import build_machine_state_snapshot


def _snapshot(generation: int) -> bytes:
    payload = bytearray(build_machine_state_snapshot((MachineExecutionState(pc=0x401000),)))
    struct.pack_into("<Q", payload, 16, generation)
    return bytes(payload)


def test_mailbox_keeps_only_complete_monotonic_snapshot_flips():
    mailbox = MachineSnapshotMailbox()

    assert mailbox.publish(_snapshot(2)) is True
    assert mailbox.publish(_snapshot(1)) is False
    assert mailbox.latest_after(1) == (2, _snapshot(2))
    assert mailbox.latest_after(2) is None

    malformed = bytearray(_snapshot(3))
    struct.pack_into("<I", malformed, 12, len(malformed) + 1)
    with pytest.raises(ValueError, match="byte size"):
        mailbox.publish(malformed)


def test_loopback_host_serves_page_flips_and_bounded_terminal_messages():
    mailbox = MachineSnapshotMailbox()
    terminal = MachineTerminalInputQueue(maximum_messages=2)
    server = build_machine_snapshot_server(
        "<html>machine</html>", mailbox, terminal,
        maximum_input_bytes=8,
    )
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f"http://{server.server_address[0]}:{server.server_address[1]}"
    try:
        with urlopen(root + "/", timeout=2) as response:
            assert response.read() == b"<html>machine</html>"

        mailbox.publish(_snapshot(7))
        with urlopen(root + "/snapshot?after=0", timeout=2) as response:
            assert response.headers["X-Turing-Snapshot-Generation"] == "7"
            assert response.read() == _snapshot(7)
        with urlopen(root + "/snapshot?after=7", timeout=2) as response:
            assert response.status == 204

        request = Request(root + "/input", data=b"dir\r\n", method="POST")
        with urlopen(request, timeout=2) as response:
            assert response.status == 204
        assert terminal.drain() == (b"dir\r\n",)

        oversized = Request(root + "/input", data=b"123456789", method="POST")
        with pytest.raises(HTTPError) as raised:
            urlopen(oversized, timeout=2)
        assert raised.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_machine_snapshot_host_rejects_network_exposure():
    with pytest.raises(ValueError, match="only to loopback"):
        build_machine_snapshot_server(
            "", MachineSnapshotMailbox(), MachineTerminalInputQueue(),
            bind="0.0.0.0",
        )


def test_live_controller_is_the_only_machine_mutator_and_publishes_flips():
    base = _snapshot(1)

    class Snapshots:
        generation = 0

        def copy_latest(self):
            self.generation += 1
            payload = bytearray(base)
            struct.pack_into("<Q", payload, 16, self.generation)
            return bytes(payload)

    class Runner:
        _last_results = ()

        def __init__(self, owner):
            self.owner = owner

        def tick(self, transitions):
            self.owner.calls.append(("tick", get_ident()))
            if transitions:
                self._last_results = (SimpleNamespace(
                    status=SimpleNamespace(name="HALTED"),
                ),)
                return 1
            return 0

    class Machine:
        def __init__(self):
            self.calls = []
            self.snapshots = Snapshots()
            self.runner = Runner(self)
            self.machine = SimpleNamespace(cores=(object(),))

        def inject_console_input(self, payload):
            self.calls.append((bytes(payload), get_ident()))

        def service_external_requests(self, _port, *, core_index):
            self.calls.append((f"service:{core_index}", get_ident()))
            return 0

        def pending_external_requests(self, _index):
            return ()

        def set_direction(self, _direction):
            self.calls.append(("direction", get_ident()))

        def service_dispatch_frontiers(self):
            self.calls.append(("dispatch", get_ident()))
            return 0

    machine = Machine()
    mailbox = MachineSnapshotMailbox()
    terminal = MachineTerminalInputQueue()
    terminal.submit(b"echo live\r\n")
    controller = LiveMachineSnapshotController(
        machine, object(), mailbox, terminal, transitions_per_cycle=4,
    )

    controller.start()
    deadline = time.monotonic() + 2
    while mailbox.generation < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    controller.stop()

    assert controller.failure is None
    assert mailbox.generation >= 2
    mutation_threads = {thread for _operation, thread in machine.calls}
    assert len(mutation_threads) == 1
    assert get_ident() not in mutation_threads
    assert (b"echo live\r\n", next(iter(mutation_threads))) in machine.calls
