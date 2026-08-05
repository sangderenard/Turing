from types import SimpleNamespace

from src.compiler.machine_state_buffer import (
    MachineRunDirection,
    SubjectOutputFormat,
    SubjectOutputKind,
)

from examples import reversible_machine_viewer as viewer


class _Runner:
    def __init__(self, *, running=True):
        self.running = running
        self.direction = MachineRunDirection.PAUSED
        self.calls = []

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def start(self, direction):
        self.calls.append(("start", direction))
        self.direction = direction
        self.running = True

    def tick(self, count):
        self.calls.append(("tick", count))
        return count


class _Machine:
    def __init__(self, *, running=True):
        self.runner = _Runner(running=running)
        self.calls = []

    def inject_console_input(self, payload):
        self.calls.append(("input", payload))

    def set_direction(self, direction):
        self.calls.append(("direction", direction))
        self.runner.direction = direction


def test_native_console_input_stops_the_worker_before_mutating_guest_state():
    machine = _Machine(running=True)

    viewer._inject_console_input(machine, b"dir\r\n", free_spin=True)

    assert machine.runner.calls == [
        "stop", ("start", MachineRunDirection.FORWARD),
    ]
    assert machine.calls == [("input", b"dir\r\n")]


def test_native_single_step_uses_runner_and_finishes_paused():
    machine = _Machine(running=True)

    viewer._single_step(machine, MachineRunDirection.BACKWARD)

    assert machine.runner.calls == ["stop", ("tick", 1)]
    assert machine.calls == [
        ("direction", MachineRunDirection.BACKWARD),
        ("direction", MachineRunDirection.PAUSED),
    ]


def test_native_forward_restarts_free_spin_after_single_step():
    machine = _Machine(running=False)

    viewer._set_native_direction(
        machine, MachineRunDirection.FORWARD, free_spin=True,
    )

    assert machine.runner.calls == [("start", MachineRunDirection.FORWARD)]
    assert machine.calls == []


def test_native_viewer_uploads_page_occupancy_as_rgba32ui(monkeypatch):
    calls = []
    for name in (
        "glActiveTexture", "glBindTexture", "glUseProgram", "glUniform2f",
    ):
        monkeypatch.setattr(viewer.gl, name, lambda *args, _name=name: calls.append((_name, args)))
    monkeypatch.setattr(
        viewer.gl, "glTexImage2D",
        lambda *args: calls.append(("glTexImage2D", args)),
    )
    payload = (
        (1).to_bytes(8, "little") + (32).to_bytes(4, "little") + (3).to_bytes(4, "little")
        + (2).to_bytes(8, "little") + (7).to_bytes(4, "little") + (9).to_bytes(4, "little")
    )
    descriptor = SimpleNamespace(
        kind=SubjectOutputKind.MEMORY_PAGES,
        format=SubjectOutputFormat.PAGE_OCCUPANCY_V1,
        width=2, height=1,
    )
    snapshot = SimpleNamespace(
        header=SimpleNamespace(
            generation=1, register_count=1, core_count=1,
            register_offset=0, register_stride_bytes=8, output_count=1,
        ),
        data=bytes(8),
        output_descriptor=lambda _index: descriptor,
        output_bytes=lambda _index: payload,
    )
    display = viewer.NativeMachineDisplay.__new__(viewer.NativeMachineDisplay)
    display.last_generation = -1
    display.register_texture = 1
    display.memory_texture = 2
    display.subject_texture = 3
    display.program = 4
    display.shape_location = 5
    display.subject_available = 0.0
    display.memory_page_count = 0

    display.upload(snapshot)

    page_uploads = [args for name, args in calls if (
        name == "glTexImage2D" and args[2] == viewer.gl.GL_RGBA32UI
    )]
    assert len(page_uploads) == 1
    assert page_uploads[0][3:5] == (2, 1)
    assert page_uploads[0][-1] == payload
    assert display.memory_page_count == 2


def test_native_viewer_selects_segmented_tape_loader(tmp_path, monkeypatch):
    tape = tmp_path / "proof.segmented-tape"
    tape.mkdir()
    sentinel = object()
    monkeypatch.setattr(
        viewer.BinaryMachineProgram, "load_segmented_system_tape",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        viewer.BinaryMachineProgram, "load_system_tape",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("JSONL loader used")),
    )
    options = SimpleNamespace(
        tape=tape, new=False, machine_backend="translated", binary=None,
    )

    assert viewer._load(options) is sentinel
