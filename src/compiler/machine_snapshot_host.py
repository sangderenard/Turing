"""Loopback transport for immutable machine-state flips and terminal input.

The executor thread is the only thread allowed to mutate a
``BinaryMachineProgram``.  HTTP request threads exchange complete TMSNAP01
buffers and bounded byte messages with it; they never receive the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from queue import Empty, Full, Queue
import socket
from threading import Condition, Event, Thread
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from .machine_state_buffer import MachineRunDirection, MachineSnapshotView


class MachineSnapshotMailbox:
    """Retain only the newest validated, monotonically numbered snapshot."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._generation = 0
        self._payload: bytes | None = None

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def publish(self, snapshot: bytes | bytearray | memoryview) -> bool:
        payload = bytes(snapshot)
        view = MachineSnapshotView(memoryview(payload))
        if view.header.byte_size != len(payload):
            raise ValueError("machine snapshot byte size does not match its header")
        with self._condition:
            if view.header.generation <= self._generation:
                return False
            self._generation = view.header.generation
            self._payload = payload
            self._condition.notify_all()
        return True

    def latest_after(
        self, generation: int, *, timeout: float | None = None,
    ) -> tuple[int, bytes] | None:
        """Return one complete newer flip, optionally waiting for publication."""

        after = int(generation)
        if after < 0:
            raise ValueError("snapshot generation cannot be negative")
        with self._condition:
            if self._generation <= after and timeout:
                self._condition.wait_for(
                    lambda: self._generation > after, timeout=float(timeout),
                )
            if self._generation <= after or self._payload is None:
                return None
            return self._generation, self._payload


class MachineTerminalInputQueue:
    """A bounded message boundary between HTTP workers and the guest owner."""

    def __init__(self, *, maximum_messages: int = 256) -> None:
        if maximum_messages <= 0:
            raise ValueError("maximum input messages must be positive")
        self._queue: Queue[bytes] = Queue(maxsize=int(maximum_messages))

    def submit(self, data: bytes | bytearray | memoryview) -> bool:
        payload = bytes(data)
        try:
            self._queue.put_nowait(payload)
        except Full:
            return False
        return True

    def drain(self) -> tuple[bytes, ...]:
        messages: list[bytes] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                return tuple(messages)


class MachineControlQueue:
    """Bounded, validated shell commands consumed only by the machine owner."""

    _ACTIONS = frozenset({
        "pause", "forward", "backward", "step_forward", "step_backward", "speed",
    })

    def __init__(self, *, maximum_messages: int = 256) -> None:
        self._queue: Queue[tuple[str, float | None]] = Queue(maxsize=maximum_messages)

    def submit(self, action: str, value: float | None = None) -> bool:
        active = str(action)
        if active not in self._ACTIONS:
            raise ValueError(f"unknown machine control action {active!r}")
        if active == "speed":
            if value is None or not 1 <= float(value) <= 10_000_000:
                raise ValueError("machine speed must be between 1 and 10000000 transitions per cycle")
        try:
            self._queue.put_nowait((active, None if value is None else float(value)))
        except Full:
            return False
        return True

    def drain(self) -> tuple[tuple[str, float | None], ...]:
        result = []
        while True:
            try:
                result.append(self._queue.get_nowait())
            except Empty:
                return tuple(result)


class MachineSubjectQueue(MachineTerminalInputQueue):
    """Bounded byte-exact binary subjects submitted by a declared file port."""


class _MachineSnapshotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        html: bytes,
        mailbox: MachineSnapshotMailbox,
        terminal_input: MachineTerminalInputQueue,
        control: MachineControlQueue,
        subjects: MachineSubjectQueue,
        *,
        maximum_input_bytes: int,
    ) -> None:
        self.html = html
        self.mailbox = mailbox
        self.terminal_input = terminal_input
        self.control = control
        self.subjects = subjects
        self.maximum_input_bytes = maximum_input_bytes
        super().__init__(server_address, _MachineSnapshotRequestHandler)


class _MachineSnapshotHTTPServerV6(_MachineSnapshotHTTPServer):
    address_family = socket.AF_INET6


class _MachineSnapshotRequestHandler(BaseHTTPRequestHandler):
    server: _MachineSnapshotHTTPServer

    def log_message(self, _format: str, *_arguments: object) -> None:
        return

    def _send(self, status: HTTPStatus, payload: bytes = b"", content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        target = urlsplit(self.path)
        if target.path == "/":
            self._send(HTTPStatus.OK, self.server.html, "text/html; charset=utf-8")
            return
        if target.path != "/snapshot":
            self._send(HTTPStatus.NOT_FOUND)
            return
        try:
            values = parse_qs(target.query, strict_parsing=False)
            after = int(values.get("after", ["0"])[-1])
            result = self.server.mailbox.latest_after(after)
        except (TypeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, b"invalid snapshot generation")
            return
        if result is None:
            self._send(HTTPStatus.NO_CONTENT)
            return
        generation, payload = result
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/vnd.turing.machine-snapshot")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Turing-Snapshot-Generation", str(generation))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlsplit(self.path).path
        if route == "/control":
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > 4096:
                    raise ValueError("invalid control length")
                value = json.loads(self.rfile.read(length))
                accepted = self.server.control.submit(value["action"], value.get("value"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send(HTTPStatus.BAD_REQUEST, str(error).encode("utf-8"))
                return
            self._send(HTTPStatus.NO_CONTENT if accepted else HTTPStatus.TOO_MANY_REQUESTS)
            return
        if route not in {"/input", "/subject"}:
            self._send(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > self.server.maximum_input_bytes:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        payload = self.rfile.read(length)
        if len(payload) != length:
            self._send(HTTPStatus.BAD_REQUEST, b"incomplete terminal input")
            return
        queue = self.server.terminal_input if route == "/input" else self.server.subjects
        if not queue.submit(payload):
            self._send(HTTPStatus.TOO_MANY_REQUESTS, b"terminal input queue is full")
            return
        self._send(HTTPStatus.NO_CONTENT)


def build_machine_snapshot_server(
    html: str | bytes,
    mailbox: MachineSnapshotMailbox,
    terminal_input: MachineTerminalInputQueue,
    *,
    bind: str = "127.0.0.1",
    port: int = 0,
    maximum_input_bytes: int = 64 * 1024,
    control: MachineControlQueue | None = None,
    subjects: MachineSubjectQueue | None = None,
) -> ThreadingHTTPServer:
    """Build a same-origin server; non-loopback exposure is rejected."""

    try:
        address = ipaddress.ip_address(bind)
    except ValueError as error:
        raise ValueError("machine snapshot host requires a numeric loopback address") from error
    if not address.is_loopback:
        raise ValueError("machine snapshot host may bind only to loopback")
    if maximum_input_bytes <= 0:
        raise ValueError("maximum input bytes must be positive")
    payload = html.encode("utf-8") if isinstance(html, str) else bytes(html)
    server_type = (
        _MachineSnapshotHTTPServerV6
        if address.version == 6 else _MachineSnapshotHTTPServer
    )
    return server_type(
        (str(address), int(port)), payload, mailbox, terminal_input,
        control or MachineControlQueue(), subjects or MachineSubjectQueue(maximum_messages=4),
        maximum_input_bytes=int(maximum_input_bytes),
    )


@dataclass(slots=True)
class LiveMachineSnapshotController:
    """Own a machine on one thread and publish its complete observable flips."""

    machine: object
    external_port: object
    mailbox: MachineSnapshotMailbox
    terminal_input: MachineTerminalInputQueue
    control: MachineControlQueue | None = None
    subjects: MachineSubjectQueue | None = None
    replacement_factory: Callable[[bytes], tuple[object, object]] | None = None
    transitions_per_cycle: int = 65_536
    idle_seconds: float = 0.004
    failure: BaseException | None = field(init=False, default=None)
    _stop: Event = field(init=False, repr=False)
    _thread: Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.transitions_per_cycle <= 0:
            raise ValueError("transitions per controller cycle must be positive")
        self._stop = Event()
        self._thread = None
        self.failure = None
        if self.control is None:
            self.control = MachineControlQueue()
        if self.subjects is None:
            self.subjects = MachineSubjectQueue(maximum_messages=4)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            raise RuntimeError("machine snapshot controller is already active")
        self.failure = None
        self._stop.clear()
        self._thread = Thread(target=self._run, name="turing-machine-owner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("machine snapshot controller did not stop")

    def _publish(self) -> None:
        snapshot = self.machine.snapshots.copy_latest()
        if snapshot is not None:
            self.mailbox.publish(snapshot)

    def _run(self) -> None:
        try:
            self.machine.runner.tick(0)
            self._publish()
            may_advance = True
            while not self._stop.is_set():
                changed = False
                single_step = False
                for action, value in self.control.drain():
                    if action == "pause":
                        self.machine.set_direction(MachineRunDirection.PAUSED)
                        may_advance = False
                    elif action in {"forward", "backward"}:
                        self.machine.set_direction(
                            MachineRunDirection.FORWARD
                            if action == "forward" else MachineRunDirection.BACKWARD
                        )
                        may_advance = True
                    elif action in {"step_forward", "step_backward"}:
                        self.machine.set_direction(
                            MachineRunDirection.FORWARD
                            if action == "step_forward" else MachineRunDirection.BACKWARD
                        )
                        may_advance = True
                        single_step = True
                    elif action == "speed":
                        self.transitions_per_cycle = int(value)
                    changed = True

                for subject in self.subjects.drain():
                    if self.replacement_factory is None:
                        raise RuntimeError("the HTML shell host did not admit binary replacement")
                    replacement, replacement_port = self.replacement_factory(subject)
                    old_machine = self.machine
                    self.machine, self.external_port = replacement, replacement_port
                    old_machine.close()
                    self.machine.runner.tick(0)
                    may_advance = False
                    changed = True
                    self._publish()

                for payload in self.terminal_input.drain():
                    self.machine.inject_console_input(payload)
                    changed = True
                    may_advance = True

                direction = getattr(
                    self.machine.runner, "direction", MachineRunDirection.FORWARD,
                )
                serviced = 0
                if direction is not MachineRunDirection.BACKWARD:
                    for core_index in range(len(self.machine.machine.cores)):
                        serviced += self.machine.service_external_requests(
                            self.external_port, core_index=core_index,
                        )
                if serviced:
                    self.machine.runner.tick(0)
                    self._publish()
                    changed = True
                    may_advance = True

                pending = any(
                    self.machine.pending_external_requests(index)
                    for index in range(len(self.machine.machine.cores))
                )
                if (direction is MachineRunDirection.BACKWARD or not pending) and may_advance:
                    transitions = self.machine.runner.tick(
                        1 if single_step else self.transitions_per_cycle
                    )
                    self._publish()
                    changed = changed or bool(transitions)
                    if single_step:
                        self.machine.set_direction(MachineRunDirection.PAUSED)
                        may_advance = False
                    results = self.machine.runner._last_results
                    at_frontier = bool(results) and any(
                        result.status.name != "RUNNING" for result in results
                    )
                    if direction is MachineRunDirection.FORWARD and (at_frontier or not transitions):
                        installed = self.machine.service_dispatch_frontiers()
                        changed = changed or bool(installed)
                        pending = any(
                            self.machine.pending_external_requests(index)
                            for index in range(len(self.machine.machine.cores))
                        )
                        may_advance = bool(installed or pending)

                if not changed:
                    self._stop.wait(self.idle_seconds)
        except BaseException as error:  # surfaced by the owning CLI/test
            self.failure = error
            self._stop.set()


def serve_machine_snapshot_host(
    server: ThreadingHTTPServer,
    controller: LiveMachineSnapshotController,
    *,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Run a controller and its loopback context server until interrupted."""

    host, port = server.server_address[:2]
    url_host = f"[{host}]" if ":" in host else host
    controller.start()
    if on_ready is not None:
        on_ready(f"http://{url_host}:{port}/")
    server.timeout = 0.1
    try:
        while controller.running and controller.failure is None:
            server.handle_request()
    finally:
        server.server_close()
        controller.stop()
    if controller.failure is not None:
        raise controller.failure


__all__ = [
    "LiveMachineSnapshotController",
    "MachineControlQueue",
    "MachineSnapshotMailbox",
    "MachineTerminalInputQueue",
    "MachineSubjectQueue",
    "build_machine_snapshot_server",
    "serve_machine_snapshot_host",
]
