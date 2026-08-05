"""Resume a real Windows AMD64 subject until its next fail-closed frontier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.compiler.binary_machine_program import BinaryMachineProgram
from src.compiler.machine_state_buffer import MachineRunDirection
from src.compiler.machine_system_ports import deterministic_windows_bootstrap_port
from src.compiler.shell_io import VirtualFileSystemContract, VirtualMount
from src.compiler.virtual_filesystem import VirtualFileSystemState
from src.compiler.virtual_filesystem import VirtualFileEffect
from src.compiler.virtual_process import VirtualProgramRegistry, VirtualProgramResult


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path(os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")))
    parser.add_argument("--tape", type=Path, default=Path("cmd-machine.tape.jsonl"))
    parser.add_argument("--new", action="store_true", help="start from the binary instead of resuming the tape")
    parser.add_argument(
        "--segmented", action="store_true",
        help="create a bounded content-addressed tape directory for a new run",
    )
    parser.add_argument(
        "--machine-backend", choices=("translated", "node-wasm"),
        default="translated",
        help="select translated Python blocks or journalled Wasm safe prefixes",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="start cmd.exe without /c and stop successfully when it awaits terminal input",
    )
    parser.add_argument(
        "--input", action="append", default=[], metavar="TEXT",
        help="journal UTF-8 terminal input before running (repeatable)",
    )
    parser.add_argument(
        "--input-line", action="append", default=[], metavar="TEXT",
        help="journal one terminal line with an appended Windows CRLF (repeatable)",
    )
    parser.add_argument(
        "--demo-card", metavar="NAME",
        help="register NAME.exe as a virtual card-set executor for process interception",
    )
    parser.add_argument("--maximum-boundaries", type=int, default=1000)
    parser.add_argument(
        "--output-tail-bytes", type=int, default=256, metavar="N",
        help="print up to N final bytes from the subject console device (zero disables)",
    )
    parser.add_argument("command", nargs="*", default=None)
    options = parser.parse_args()
    if options.output_tail_bytes < 0:
        parser.error("--output-tail-bytes cannot be negative")
    command = () if options.interactive else tuple(options.command or ("/c", "echo hello"))

    environment = {
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    }
    machine_backend = (
        None if options.machine_backend == "translated" else options.machine_backend
    )
    if options.tape.exists() and not options.new:
        if options.segmented and not options.tape.is_dir():
            parser.error("--segmented resume requires a tape directory")
        machine = (
            BinaryMachineProgram.load_segmented_system_tape(
                options.tape, maximum_file_size=128 * 1024 * 1024,
                machine_block_backend=machine_backend,
            )
            if options.tape.is_dir()
            else BinaryMachineProgram.load_system_tape(
                options.tape, maximum_file_size=128 * 1024 * 1024,
                machine_block_backend=machine_backend,
            )
        )
    else:
        subject = options.binary.read_bytes()
        filesystem = VirtualFileSystemState.create(
            VirtualFileSystemContract(
                current_directory="/c/work",
                mounts=(VirtualMount.create("/", "memory", access="read_write"),),
            ),
            files={"/c/windows/system32/cmd.exe": subject},
        )
        machine = BinaryMachineProgram.load_pe(
            subject, maximum_file_size=128 * 1024 * 1024,
            virtual_filesystem=filesystem, virtual_environment=environment,
            machine_block_backend=machine_backend,
        )
        if options.segmented:
            machine.begin_segmented_system_tape(options.tape)

    program_registry = None
    if options.demo_card:
        program_registry = VirtualProgramRegistry()
        card_path = f"/c/work/{options.demo_card}.exe"
        program_registry.register(
            card_path,
            bundle_reference=f"bundle:demo/{options.demo_card}@local",
            executor_reference=f"card-set:{options.demo_card}:v1",
            executor=lambda invocation: VirtualProgramResult(
                0,
                (f"[card-set:{options.demo_card}] " + " ".join(invocation.arguments) + "\r\n").encode(),
                execution_units=max(1, len(invocation.arguments)),
            ),
        )
        filesystem = machine.machine.cores[0].state.virtual_filesystem
        try:
            existing_card = filesystem.stat(card_path) if filesystem is not None else None
        except FileNotFoundError:
            machine.apply_shell_file_effect(VirtualFileEffect(
                "create", card_path,
                # cmd performs executable-format probing before CreateProcess.
                # The registry still owns execution; these admitted bytes are
                # only a recognizable PE-shaped VFS marker.
                data=machine.system_tape.subject_binary,
            ))
        else:
            if existing_card is not None and existing_card.data[:2] != b"MZ":
                machine.apply_shell_file_effect(VirtualFileEffect(
                    "write", card_path,
                    data=machine.system_tape.subject_binary,
                ))

    port = deterministic_windows_bootstrap_port(
        arguments=("cmd.exe", *command),
        environment=tuple(f"{key}={value}" for key, value in environment.items()),
        module_virtual_path="/c/windows/system32/cmd.exe",
        program_registry=program_registry,
    )
    serviced = 0
    outcome = "boundary limit"
    try:
        for terminal_input in (*options.input, *(item + "\r\n" for item in options.input_line)):
            machine.inject_console_input(terminal_input)
        for _ in range(options.maximum_boundaries):
            if machine.machine.cores[0].state.halted:
                state = machine.machine.cores[0].state
                outcome = (
                    f"HALTED at step {state.steps}: resumed exact exit code "
                    f"{int(state.exit_code or 0)}"
                )
                break
            if machine.pending_external_requests():
                count = machine.service_external_requests(port)
                serviced += count
                if not count:
                    request = machine.pending_external_requests()[0]
                    reference_name = (
                        f"{request.reference.library}!{request.reference.symbol}"
                    )
                    if port.wait_kind(request) == "terminal_input":
                        outcome = (
                            f"RECEPTIVE at step {machine.machine.cores[0].state.steps}: "
                            f"{reference_name} is waiting for shell input"
                        )
                        machine.annotate_tape(
                            "terminal_receptive", outcome,
                            color="cyan", severity="breakpoint", core=0,
                            position=machine.machine.cores[0].position,
                            address=request.instruction_address,
                            external_reference=reference_name,
                            metadata={"status": "WAITING_INPUT"},
                        )
                        break
                    outcome = (
                        f"unsupported external at step {machine.machine.cores[0].state.steps}: "
                        f"{reference_name} "
                        f"arguments={request.arguments} stack={request.stack_arguments}"
                    )
                    machine.annotate_tape(
                        "unsupported_external", outcome,
                        color="amber", severity="caution", core=0,
                        position=machine.machine.cores[0].position,
                        address=request.instruction_address,
                        external_reference=reference_name,
                        metadata={"status": "WAITING_EXTERNAL"},
                    )
                    break
            machine.set_direction(MachineRunDirection.FORWARD)
            machine.runner.tick(1_000_000)
            if not machine.pending_external_requests():
                result = machine.runner._last_results[0]
                if (
                    result.status.name == "BLOCKED_CONTROL"
                    and machine.service_dispatch_frontiers(core_index=0)
                ):
                    continue
                if result.status.name == "HALTED":
                    outcome = (
                        f"HALTED at step {result.state.steps}: {result.reason}"
                    )
                    machine.annotate_tape(
                        "process_exit", outcome,
                        color="green", severity="verified", core=0,
                        position=machine.machine.cores[0].position,
                        address=result.state.pc,
                        metadata={
                            "status": "HALTED",
                            "exit_code": result.state.exit_code,
                        },
                    )
                    break
                if result.status.name != "RUNNING":
                    outcome = f"{result.status.name} at step {result.state.steps}: {result.reason}"
                    machine.annotate_tape(
                        "execution_frontier", outcome,
                        color="red", severity="error", core=0,
                        position=machine.machine.cores[0].position,
                        address=result.state.pc,
                        metadata={"status": result.status.name},
                    )
                    break
    except Exception as error:
        machine.annotate_tape(
            "emulator_exception", f"{type(error).__name__}: {error}",
            color="magenta", severity="suspect", core=0,
            position=machine.machine.cores[0].position,
            address=machine.machine.cores[0].state.pc,
        )
        machine.close()
        raise
    finally:
        machine.save_system_tape(options.tape)
        if program_registry is not None:
            program_registry.export_child_tapes(
                options.tape.parent / f"{options.tape.name}.children",
            )
    print(outcome)
    print(
        f"serviced={serviced} tape_records={len(machine.system_tape.records)} "
        f"annotations={len(machine.system_tape.annotations)} tape={options.tape.resolve()}"
    )
    if machine.recompilation_statistics:
        print("recompilation=" + repr(dict(machine.recompilation_statistics)))
    output = machine.machine.cores[0].state.device_state.get("console.output", b"")
    if output and options.output_tail_bytes:
        print("subject_output_tail=" + repr(output[-options.output_tail_bytes:]))
    if machine.system_tape.annotations:
        annotation = machine.system_tape.annotations[-1]
        print(
            f"flag={annotation.color}/{annotation.severity} "
            f"feature={annotation.feature} sequence={annotation.sequence_start} "
            f"rip={annotation.address!r}"
        )
    machine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
