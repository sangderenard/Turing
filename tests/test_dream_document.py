from pathlib import Path
from threading import Barrier

import pytest

from src.compiler.dream_document import (
    DreamDocumentError,
    DreamRuntime,
    load_dream_document,
    parse_dream_document,
    python_exec_handler,
    main,
)
from src.compiler.wasm_html_shell import emit_html_shell


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

    assert namespace["chip_state"]["cycle"] == 1
    assert namespace["register_layout"].core_stride == 256
    assert records[1].result == 1


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
    assert "head-step: python -> 1" in output
