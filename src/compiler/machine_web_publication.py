"""Build a common static HTML-shell publication from a reversible machine."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from .dream_document import embed_machine_snapshot_replay, embed_machine_wasm_block_bootstrap
from .machine_block_web_bundle import build_machine_block_web_bundle
from .machine_state_buffer import MachineRunDirection
from .wasm_html_shell import HtmlShell


@dataclass(frozen=True, slots=True)
class MachineWebPublication:
    """Common page, assets, and runtime manifest ready for a site bundle."""

    html: str
    assets: Mapping[str, bytes]
    runtime: Mapping[str, Any]


def build_machine_web_publication(
    shell: HtmlShell,
    machine,
    *,
    document_source: bytes,
    subject: bytes,
    subject_path: str,
    subject_metadata: Mapping[str, Any] | None = None,
) -> MachineWebPublication:
    """Project one machine's entry block through the shared static shell ABI.

    The retained executor frames are a fail-safe replay. On a capable browser,
    the authenticated register-only Wasm journal replaces them with every
    computed checkpoint through the standard ``TMSNAP01`` transport.
    """

    core = machine.machine.cores[0]
    recompiled = core.executor.recompile_block_wasm(
        core.state.pc, core.state, strict=False,
    )
    subject_digest = sha256(subject).hexdigest()
    block = build_machine_block_web_bundle(
        recompiled, core.state, subject_sha256=subject_digest,
    )

    machine.runner.tick(0)
    initial = machine.snapshots.copy_latest()
    replay_frames = [] if initial is None else [initial]
    machine.runner.set_direction(MachineRunDirection.FORWARD)
    machine.runner.tick(1)
    advanced = machine.snapshots.copy_latest()
    if advanced is not None:
        replay_frames.append(advanced)
    machine.runner.set_direction(MachineRunDirection.PAUSED)

    published = embed_machine_snapshot_replay(shell, replay_frames)
    published = embed_machine_wasm_block_bootstrap(published, block.descriptor)
    html = "\n".join(line.rstrip() for line in published.html.splitlines())
    if published.html.endswith(("\n", "\r")):
        html += "\n"

    published_subject = {
        "path": subject_path,
        "sha256": subject_digest,
        **dict(subject_metadata or {}),
    }
    runtime = {
        "schema": "turing.reversible-machine-runtime.v1",
        "document_digest": sha256(document_source).hexdigest(),
        "shell_digest": sha256(html.encode("utf-8")).hexdigest(),
        "display_owner": "program-interior",
        "system_ports": {
            "subject": "/subject",
            "terminal_input": "/input",
            "machine_control": "/control",
            "snapshots": "/snapshot",
        },
        "controls": [
            "pause", "forward", "backward", "step_forward", "step_backward", "speed",
        ],
        "snapshot_abi": "TMSNAP01",
        "memory_page_bytes": 4096,
        "static_preview": initial is not None,
        "static_replay_frames": len(replay_frames),
        "published_subject": published_subject,
        "recompiled_machine_block": {
            **dict(block.descriptor),
            "complete": bool(block.plan["complete"]),
            "covered_operation_count": int(block.plan["covered_operation_count"]),
            "shortfalls": list(block.plan["shortfalls"]),
        },
    }
    return MachineWebPublication(
        html=html,
        assets={subject_path: bytes(subject), **block.assets},
        runtime=runtime,
    )


__all__ = ["MachineWebPublication", "build_machine_web_publication"]
