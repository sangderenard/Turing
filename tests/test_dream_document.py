import base64
from pathlib import Path
from threading import Barrier

import pytest

from src.compiler.dream_document import (
    DreamDocumentError,
    DreamRuntime,
    embed_machine_snapshot,
    embed_machine_snapshot_replay,
    embed_machine_snapshot_stream,
    emit_dream_html_shell,
    load_dream_document,
    parse_dream_document,
    python_exec_handler,
    main,
)
from src.compiler.wasm_html_shell import HtmlShell, emit_html_shell


DOCUMENT = Path(__file__).parents[1] / "examples" / "reversible_chip_simulator.dream"


def test_reversible_chip_dream_document_parses_every_language_arena():
    document = load_dream_document(DOCUMENT)

    assert [block.identity for block in document.blocks] == [
        "chip-setup", "head-step", "register-light-compute",
        "chip-present-fragment", "gpu-indicator-ui",
    ]
    assert [block.language for block in document.blocks] == [
        "python", "python", "glsl", "glsl", "javascript",
    ]
    assert document.block("register-light-compute").stage == "compute"
    assert document.block("chip-present-fragment").stage == "fragment"
    assert document.parallel[0].members == (
        "head-step", "register-light-compute",
    )


def test_dream_document_projects_to_lazy_card_graph_with_parallel_deployment():
    graph = load_dream_document(DOCUMENT).card_graph()

    assert graph["abi"] == "turing.card-graph.v1"
    assert graph["document_schema"] == "turing.dream-document.v1"
    assert graph["paths"]["linear"] == [
        "chip-setup", "head-step", "register-light-compute",
        "chip-present-fragment", "gpu-indicator-ui",
    ]
    assert graph["parallel_deployments"] == [{
        "id": "chip-frame",
        "members": ["head-step", "register-light-compute"],
        "join": "frame-barrier",
    }]
    assert graph["block_metadata"]["register-light-compute"]["gpu_deployment"] is True


def test_dream_deployments_survive_as_catalogued_ssa_dispatch_regions():
    lowered = load_dream_document(DOCUMENT).lower_to_ssa()
    regions = lowered.module.deployment_table["dream_main"]

    assert [region.kind for region in regions] == [
        "parallel_deployment", "device_dispatch", "device_dispatch",
    ]
    assert regions[0].schedule == "independent_lanes"
    assert regions[0].lanes[0].callees == ("dream_block_head_step",)
    assert regions[0].lanes[1].callees == ("dream_block_register_light_compute",)
    assert regions[1].schedule == "glsl:compute"
    assert regions[2].schedule == "glsl:fragment"
    assert lowered.module.functions["dream_main"].metadata["deployment_regions"] == regions


def test_shell_hands_context_to_interior_display_owner_without_compiling_it():
    document = load_dream_document(DOCUMENT)
    handoff = document.display_handoff()
    assert handoff is not None
    assert handoff.owner == "chip-present-fragment"
    assert handoff.context == "webgl2"
    assert "void main()" in handoff.fragment_source
    assert handoff.controller_entry == "installTuringDisplay"
    assert "terminalForm.dataset.turingTerminalInput" in handoff.controller_source

    api = {
        "module": "dream-chip", "language": "dream", "entry": "run",
        "entry_points": [{"name": "run", "symbol": "run", "parameters": []}],
        "metadata": {},
    }
    html = emit_html_shell(
        api,
        shader_execution=handoff.to_shader_execution(),
        map_ir={"card_graph": document.card_graph()},
    ).html

    assert 'SHADER.display_ownership === "program-interior"' in html
    assert "The shell stops here" in html
    assert "interior.controller_source" in html
    assert "installTuringDisplay" in html
    assert 'canvas.dataset.displayOwner = interior.owner' in html

    complete = emit_dream_html_shell(document, name="chip").html
    assert '"display_ownership": "program-interior"' in complete
    assert "TuringMachineSnapshots" in complete
    assert "machineSnapshots.subscribe(uploadSnapshot)" in complete
    assert "BinaryMachineProgram.load_pe" in complete
    assert "systemPorts.publishFile" in complete
    assert '"kind": "device_dispatch"' in complete
    assert "installTuringDisplay" in complete
    assert 'kind === 2 && format === 2' in complete
    assert 'new TextDecoder("utf-8"' in complete
    assert "uploadTerminal" in complete
    assert "turingRegisterHud" in complete
    assert "Contiguous virtual register contents" in complete
    assert "value.toString(16).padStart(16" in complete


def test_parallel_blocks_really_overlap_without_a_runtime_state_lock():
    document = load_dream_document(DOCUMENT)
    barrier = Barrier(2, timeout=2)
    indicator = []

    def python(block):
        if block.identity == "head-step":
            barrier.wait()
        return block.identity

    def shader(block):
        if block.identity == "register-light-compute":
            barrier.wait()
        return f"deployed:{block.stage}"

    records = DreamRuntime(document).run(
        {"python": python, "javascript": lambda block: block.identity},
        shader_deployer=shader,
        gpu_indicator=lambda active, block: indicator.append((active, block.identity)),
    )

    assert [record.block for record in records] == [
        "chip-setup", "head-step", "register-light-compute",
        "chip-present-fragment", "gpu-indicator-ui",
    ]
    assert indicator == [
        (True, "register-light-compute"),
        (False, "register-light-compute"),
        (True, "chip-present-fragment"),
        (False, "chip-present-fragment"),
    ]


def test_trusted_python_blocks_share_the_simulator_arena():
    document = load_dream_document(DOCUMENT)
    namespace = {}
    records = DreamRuntime(document).run(
        {
            "python": python_exec_handler(namespace),
            "javascript": lambda _block: None,
        },
        shader_deployer=lambda block: block.stage,
    )

    assert namespace["machine_program"] is None
    assert set(namespace["machine_controls"]) == {"load_binary", "tick", "set_speed"}
    assert namespace["published_generation"] == 0
    assert records[1].result == 0


def test_unframed_source_and_bad_hash_fail_before_any_language_parser():
    with pytest.raises(DreamDocumentError, match="every non-whitespace byte"):
        parse_dream_document("ordinary source\n/*@turing.segment.v1\nid=x\nlanguage=python\n@end*/\npass\n/*@turing.end*/")

    with pytest.raises(DreamDocumentError, match="failed its sha256"):
        parse_dream_document(
            "/*@turing.segment.v1\nid=x\nlanguage=python\nsha256=deadbeef\n@end*/\npass\n/*@turing.end*/"
        )


def test_reference_cli_runs_card_order_and_reports_gpu_activity(capsys):
    assert main([str(DOCUMENT), "--run-reference"]) == 0
    output = capsys.readouterr().out

    assert "GPU ACTIVE | register-light-compute" in output
    assert "GPU IDLE | chip-present-fragment" in output
    assert "head-step: python -> 0" in output


def test_cli_emits_launchable_interior_owned_shell(tmp_path, capsys):
    output = tmp_path / "chip.html"
    assert main([str(DOCUMENT), "--emit-shell", str(output)]) == 0
    html = output.read_text(encoding="utf-8")

    assert '<canvas id="shader-surface"' in html
    assert "installTuringDisplay" in html
    assert "program-interior" in html
    assert str(output) in capsys.readouterr().out


def test_machine_snapshot_can_boot_inside_generated_shell():
    artifact = HtmlShell("chip", "<html><body>chip</body></html>", False)
    snapshot = b"TMSNAP01" + bytes(68)

    embedded = embed_machine_snapshot(artifact, snapshot)

    assert embedded.name == artifact.name
    assert 'id="turing-embedded-machine-snapshot"' in embedded.html
    assert "TuringMachineSnapshots.publish(snapshot)" in embedded.html
    assert base64.b64encode(snapshot).decode("ascii") in embedded.html


def test_machine_snapshot_embedding_rejects_unframed_state():
    artifact = HtmlShell("chip", "<html><body></body></html>", False)
    with pytest.raises(ValueError, match="TMSNAP01"):
        embed_machine_snapshot(artifact, b"not-a-snapshot")


def test_machine_snapshot_replay_embeds_static_forward_and_backward_controls():
    artifact = HtmlShell("chip", "<html><body></body></html>", False)
    frames = [b"TMSNAP01" + bytes(68), b"TMSNAP01" + bytes([1]) + bytes(67)]

    embedded = embed_machine_snapshot_replay(artifact, frames)

    assert 'id="turing-embedded-machine-replay"' in embedded.html
    assert "embedded-replay" in embedded.html
    assert 'action === "step_backward"' in embedded.html
    assert "api.localReplay" in embedded.html
    assert all(base64.b64encode(frame).decode("ascii") in embedded.html for frame in frames)


def test_live_machine_snapshot_stream_boots_same_origin_transport():
    artifact = HtmlShell("chip", "<html><body></body></html>", False)

    embedded = embed_machine_snapshot_stream(
        artifact, snapshot_endpoint="/state", input_endpoint="/terminal",
        interval_milliseconds=8,
    )

    assert 'id="turing-live-machine-snapshot"' in embedded.html
    assert 'TuringMachineSnapshots.connect("/state"' in embedded.html
    assert 'inputEndpoint: "/terminal"' in embedded.html
    assert 'controlEndpoint: "/control"' in embedded.html
    assert 'subjectEndpoint: "/subject", interval: 8' in embedded.html
