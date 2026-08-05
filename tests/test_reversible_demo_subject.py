from hashlib import sha256

from examples.reversible_demo_subject import DEMO_ENTRY_CODE, build_reversible_demo_subject
from examples.reversible_machine_web_host import _new_machine_bytes
from src.compiler.machine_state_buffer import MachineRunDirection
from src.compiler.machine_web_publication import build_machine_web_publication
from src.compiler.wasm_html_shell import HtmlShell


def test_demo_subject_recompiles_and_reverses_a_multi_instruction_prefix():
    subject = build_reversible_demo_subject()
    assert subject[0x400:0x400 + len(DEMO_ENTRY_CODE)] == DEMO_ENTRY_CODE

    machine = _new_machine_bytes(subject, {}, "translated")
    try:
        core = machine.machine.cores[0]
        entry = core.state.pc
        initial_rax = core.state.registers[0]
        block = core.executor.recompile_block_wasm(entry, core.state, strict=False)

        assert [witness.semantic for witness in block.witnesses] == [
            "REGISTER_OR_MEMORY_WRITE",
            "INTEGER_ADD",
            "NO_OPERATION",
        ]
        assert [witness.encoded for witness in block.witnesses] == [
            DEMO_ENTRY_CODE[:7], DEMO_ENTRY_CODE[7:11], DEMO_ENTRY_CODE[11:12],
        ]
        assert block.covered_operation_count == 3
        assert len(block.shortfalls) == 1
        assert "call/return" in block.shortfalls[0].reason

        machine.runner.set_direction(MachineRunDirection.FORWARD)
        assert machine.runner.tick(3) == 3
        assert core.state.registers[0] == 43
        assert core.state.pc == entry + 12
        assert core.state.steps == 3

        machine.runner.set_direction(MachineRunDirection.BACKWARD)
        assert machine.runner.tick(3) == 3
        assert core.state.registers[0] == initial_rax
        assert core.state.pc == entry
        assert core.state.steps == 0
    finally:
        machine.close()


def test_common_machine_web_publication_owns_preview_assets_and_shell_contract():
    subject = build_reversible_demo_subject()
    machine = _new_machine_bytes(subject, {}, "translated")
    try:
        publication = build_machine_web_publication(
            HtmlShell("chip", "<html><body></body></html>", False),
            machine,
            document_source=b"dream source",
            subject=subject,
            subject_path="subject/demo.pe",
            subject_metadata={"format": "PE32+", "isa": "AMD64"},
        )
    finally:
        machine.close()

    runtime = publication.runtime
    assert publication.assets["subject/demo.pe"] == subject
    assert publication.assets["machine/recompiled-entry/block.wasm"].startswith(b"\0asm")
    assert runtime["published_subject"] == {
        "path": "subject/demo.pe",
        "sha256": sha256(subject).hexdigest(),
        "format": "PE32+",
        "isa": "AMD64",
    }
    assert runtime["recompiled_machine_block"]["covered_operation_count"] == 3
    assert runtime["static_replay_frames"] == 5
    assert runtime["static_replay_complete"] is True
    assert "system_ports" not in runtime  # dead decoration, deliberately removed
    assert "controls" not in runtime  # same: never read by anything
    # Terminal input / machine control / snapshot delivery's real, working
    # machinery -- not a "system_ports" dict entry.
    assert "TuringMachineSnapshots" in publication.html
    assert "TuringRecompiledMachineBlock" in publication.html
    assert "turing-embedded-machine-replay" in publication.html
    assert "copyFrames()" in publication.html
    assert "retained.slice(projected.length)" in publication.html
