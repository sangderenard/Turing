"""Run a reversible AMD64 subject behind its live dream-document display."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import webbrowser

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.dream_document import (
    embed_machine_snapshot_replay,
    embed_machine_snapshot_stream,
    emit_dream_html_shell,
    load_dream_document,
)
from src.compiler.machine_snapshot_host import (
    LiveMachineSnapshotController,
    MachineControlQueue,
    MachineSnapshotMailbox,
    MachineSubjectQueue,
    MachineTerminalInputQueue,
    build_machine_snapshot_server,
    serve_machine_snapshot_host,
)
from src.compiler.machine_state_buffer import MachineRunDirection
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port
from src.compiler.site_bundle import publish_prebuilt_program_bundle
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import VirtualFileEffect, VirtualFileSystemState
from src.compiler.virtual_process import VirtualProgramRegistry, VirtualProgramResult
from examples.reversible_demo_subject import build_reversible_demo_subject


def _new_machine(
    binary: Path, environment: dict[str, str], machine_backend: str,
) -> BinaryMachineProgram:
    return _new_machine_bytes(binary.read_bytes(), environment, machine_backend)


def _new_machine_bytes(
    subject: bytes, environment: dict[str, str], machine_backend: str,
) -> BinaryMachineProgram:
    filesystem = VirtualFileSystemState.create(
        VirtualFileSystemContract(
            current_directory="/c/work",
            mounts=(VirtualMount.create("/", "memory", access="read_write"),),
        ),
        files={"/c/windows/system32/cmd.exe": subject},
    )
    return BinaryMachineProgram.load_pe(
        subject, maximum_file_size=128 * 1024 * 1024,
        virtual_filesystem=filesystem, virtual_environment=environment,
        machine_block_backend=(None if machine_backend == "translated" else machine_backend),
    )


def _install_demo_card(
    machine: BinaryMachineProgram, name: str | None,
) -> VirtualProgramRegistry | None:
    if not name:
        return None
    registry = VirtualProgramRegistry()
    card_path = f"/c/work/{name}.exe"
    registry.register(
        card_path,
        bundle_reference=f"bundle:demo/{name}@local",
        executor_reference=f"card-set:{name}:v1",
        executor=lambda invocation: VirtualProgramResult(
            0,
            (f"[card-set:{name}] " + " ".join(invocation.arguments) + "\r\n").encode(),
            execution_units=max(1, len(invocation.arguments)),
        ),
    )
    filesystem = machine.machine.cores[0].state.virtual_filesystem
    try:
        existing = filesystem.stat(card_path) if filesystem is not None else None
    except FileNotFoundError:
        existing = None
    if existing is None:
        machine.apply_shell_file_effect(VirtualFileEffect(
            "create", card_path, data=machine.system_tape.subject_binary,
        ))
    elif existing.data[:2] != b"MZ":
        machine.apply_shell_file_effect(VirtualFileEffect(
            "write", card_path, data=machine.system_tape.subject_binary,
        ))
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path,
        default=Path(os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")),
    )
    parser.add_argument(
        "--tape", type=Path,
        default=Path("build/live-machine.segmented-tape"),
    )
    parser.add_argument("--new", action="store_true", help="start from the subject binary")
    parser.add_argument(
        "--document", type=Path,
        default=PACKAGE_ROOT / "examples" / "reversible_chip_simulator.dream",
    )
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--transitions-per-cycle", type=int, default=65_536)
    parser.add_argument(
        "--machine-backend", choices=("translated", "node-wasm"),
        default="translated",
        help="select translated Python blocks or automatic journalled Wasm safe prefixes",
    )
    parser.add_argument("--demo-card", metavar="NAME")
    parser.add_argument("--open", action="store_true", help="open the display in a browser")
    parser.add_argument(
        "--publish-bundle", type=Path, metavar="DIRECTORY",
        help="publish a common versioned program bundle beneath this website root",
    )
    parser.add_argument(
        "--publish-only", action="store_true",
        help="write --publish-bundle and exit without starting the loopback runtime",
    )
    parser.add_argument("command", nargs="*", default=None)
    options = parser.parse_args(argv)

    environment = {
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    }
    document = load_dream_document(options.document)
    artifact = embed_machine_snapshot_stream(emit_dream_html_shell(document))
    if options.publish_only and options.publish_bundle is None:
        parser.error("--publish-only requires --publish-bundle")
    if options.publish_bundle is not None:
        demo_subject = build_reversible_demo_subject()
        preview_machine = _new_machine_bytes(
            demo_subject, environment, options.machine_backend,
        )
        preview_machine.runner.tick(0)
        preview = preview_machine.snapshots.copy_latest()
        replay_frames = [] if preview is None else [preview]
        preview_machine.runner.set_direction(MachineRunDirection.FORWARD)
        preview_machine.runner.tick(1)
        advanced = preview_machine.snapshots.copy_latest()
        if advanced is not None:
            replay_frames.append(advanced)
        published_artifact = embed_machine_snapshot_replay(artifact, replay_frames)
        published_html = "\n".join(
            line.rstrip() for line in published_artifact.html.splitlines()
        )
        if published_artifact.html.endswith(("\n", "\r")):
            published_html += "\n"
        fixture_source = Path(__file__).with_name("reversible_demo_subject.py")
        bundle = publish_prebuilt_program_bundle(
            destination=options.publish_bundle,
            slug="reversible-binary-machine",
            title="Reversible Binary Machine",
            entrypoint="load_subject",
            html=published_html,
            source_filename=f"dream/{options.document.name}",
            source=options.document.read_bytes(),
            artifacts={
                f"source/python_source/{fixture_source.name}": fixture_source.read_bytes(),
                "subject/reversible-demo-amd64.pe": demo_subject,
            },
            runtime={
            "schema": "turing.reversible-machine-runtime.v1",
            "document_digest": sha256(options.document.read_bytes()).hexdigest(),
            "shell_digest": sha256(published_html.encode("utf-8")).hexdigest(),
            "display_owner": "program-interior",
            "system_ports": {
                "subject": "/subject", "terminal_input": "/input",
                "machine_control": "/control", "snapshots": "/snapshot",
            },
            "controls": [
                "pause", "forward", "backward", "step_forward", "step_backward", "speed",
            ],
            "snapshot_abi": "TMSNAP01",
            "memory_page_bytes": 4096,
            "static_preview": preview is not None,
            "static_replay_frames": len(replay_frames),
            "published_subject": {
                "path": "subject/reversible-demo-amd64.pe",
                "format": "PE32+", "isa": "AMD64",
                "sha256": sha256(demo_subject).hexdigest(),
                "entry_code_hex": demo_subject[0x400:0x402].hex(),
                "license": "project-authored fixture",
            },
            },
            refresh_gallery=True,
        )
        preview_machine.close()
        print(f"published HTML shell bundle: {bundle.directory.resolve()}", flush=True)
        print(f"public route: {bundle.url}", flush=True)
        if options.publish_only:
            return 0
    if options.tape.exists() and not options.new:
        machine = (
            BinaryMachineProgram.load_segmented_system_tape(
                options.tape, maximum_file_size=128 * 1024 * 1024,
                machine_block_backend=(
                    None if options.machine_backend == "translated" else options.machine_backend
                ),
            )
            if options.tape.is_dir()
            else BinaryMachineProgram.load_system_tape(
                options.tape, maximum_file_size=128 * 1024 * 1024,
                machine_block_backend=(
                    None if options.machine_backend == "translated" else options.machine_backend
                ),
            )
        )
    else:
        machine = _new_machine(options.binary, environment, options.machine_backend)
        machine.begin_segmented_system_tape(options.tape)

    registry = _install_demo_card(machine, options.demo_card)
    command = tuple(options.command or ())
    port = deterministic_windows_bootstrap_port(
        arguments=("cmd.exe", *command),
        environment=tuple(f"{key}={value}" for key, value in environment.items()),
        current_directory=r"C:\work",
        module_virtual_path="/c/windows/system32/cmd.exe",
        program_registry=registry,
    )
    mailbox = MachineSnapshotMailbox()
    terminal = MachineTerminalInputQueue()
    controls = MachineControlQueue()
    subjects = MachineSubjectQueue(maximum_messages=4)
    server = build_machine_snapshot_server(
        artifact.html, mailbox, terminal, bind=options.bind, port=options.port,
        maximum_input_bytes=128 * 1024 * 1024,
        control=controls, subjects=subjects,
    )

    def replacement_factory(subject: bytes):
        replacement = _new_machine_bytes(subject, environment, options.machine_backend)
        replacement_registry = _install_demo_card(replacement, options.demo_card)
        replacement_port = deterministic_windows_bootstrap_port(
            arguments=("subject.exe",),
            environment=tuple(f"{key}={value}" for key, value in environment.items()),
            current_directory=r"C:\work",
            module_virtual_path="/c/windows/system32/cmd.exe",
            program_registry=replacement_registry,
        )
        return replacement, replacement_port

    controller = LiveMachineSnapshotController(
        machine, port, mailbox, terminal,
        control=controls, subjects=subjects,
        replacement_factory=replacement_factory,
        transitions_per_cycle=options.transitions_per_cycle,
    )

    def ready(url: str) -> None:
        print(f"reversible machine display: {url}", flush=True)
        print(
            "terminal input API: await TuringMachineSnapshots.sendTerminalInput('dir\\r\\n')",
            flush=True,
        )
        print(
            "controls: TuringMachineSnapshots.sendControl('pause'|'forward'|'backward'|'step_forward'|'step_backward')",
            flush=True,
        )
        if options.open:
            webbrowser.open(url)

    try:
        serve_machine_snapshot_host(server, controller, on_ready=ready)
    except KeyboardInterrupt:
        pass
    finally:
        active_machine = controller.machine
        if active_machine is machine:
            machine.system_tape.flush() if hasattr(machine.system_tape, "flush") else machine.save_system_tape(options.tape)
        active_machine.close()
    if controller.failure is not None:
        raise controller.failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
