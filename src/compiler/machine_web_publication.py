"""Build a common static HTML-shell publication from a reversible machine."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from .dream_document import embed_machine_snapshot_replay, embed_machine_wasm_block_bootstrap
from .machine_block_web_bundle import build_machine_block_web_bundle
from .machine_state_buffer import MachineRunDirection
from .machine_system_ports import CapabilityGatedExternalPort
from .machine_tape_forwarding import TapeForwardingExternalPort
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
    maximum_replay_transitions: int = 4096,
    external_port: CapabilityGatedExternalPort | None = None,
) -> MachineWebPublication:
    """Project one machine's entry block through the shared static shell ABI.

    The retained executor frames are a fail-safe replay. On a capable browser,
    the authenticated register-only Wasm journal replaces them with every
    computed checkpoint through the standard ``TMSNAP01`` transport.

    A tick that halts on an external-call boundary is retried once against
    ``TapeForwardingExternalPort(tape=machine.system_tape, live=external_port)``:
    ``external_port`` services it live if the caller supplied one covering
    that capability, otherwise the exact completion is reused from
    ``machine.system_tape`` if this exact request was already recorded there
    by a prior native run. Neither path executes anything the machine has
    not already reversibly recorded or an explicitly supplied handler
    admits; a boundary neither can resolve still stops replay generation
    there, same as before this parameter existed.

    Terminal input, machine control (pause/forward/backward/step/speed),
    and snapshot delivery are not declared here as ports of any kind --
    they already have real, working, wired-up JS machinery
    (``window.TuringMachineSnapshots`` in wasm_html_shell.py, overridden
    for pure client-side replay by ``embed_machine_snapshot_replay`` and
    ``embed_machine_wasm_block_bootstrap`` below). The endpoint-name
    strings this function used to also put in ``runtime["system_ports"]``/
    ``runtime["controls"]`` were never read by anything that consumes this
    manifest (confirmed: every real reader of a "system_ports" key reads
    it from the ShellIOManifest-derived
    ``metadata["shell_io"]["requirements"]["system_ports"]`` path, not
    from this dict) -- dead decoration describing a live-native-host HTTP
    transport (``embed_machine_snapshot_stream``) that this function does
    not even use. Removed rather than migrated: there was nothing real to
    migrate.
    """

    core = machine.machine.cores[0]
    recompiled = core.executor.recompile_block_wasm(
        core.state.pc, core.state, strict=False,
    )
    subject_digest = sha256(subject).hexdigest()
    block = build_machine_block_web_bundle(
        recompiled, core.state, subject_sha256=subject_digest,
    )

    if maximum_replay_transitions <= 0:
        raise ValueError("machine web replay transition limit must be positive")

    machine.runner.tick(0)
    initial = machine.snapshots.copy_latest()
    replay_frames = [] if initial is None else [initial]
    machine.runner.set_direction(MachineRunDirection.FORWARD)
    replay_complete = False
    forwarding_port = TapeForwardingExternalPort(
        tape=machine.system_tape, live=external_port,
    )
    for _transition in range(maximum_replay_transitions):
        completed = machine.runner.tick(1)
        if not completed and core.state.external_requests:
            serviced = machine.service_external_requests(forwarding_port, core_index=0)
            if serviced:
                machine.runner.set_direction(MachineRunDirection.FORWARD)
                completed = machine.runner.tick(1)
        if completed:
            advanced = machine.snapshots.copy_latest()
            if advanced is not None:
                replay_frames.append(advanced)
        if machine.runner.direction is MachineRunDirection.PAUSED:
            replay_complete = True
            break
        if not completed:
            break
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
        "snapshot_abi": "TMSNAP01",
        "memory_page_bytes": 4096,
        "static_preview": initial is not None,
        "static_replay_frames": len(replay_frames),
        "static_replay_complete": replay_complete,
        "static_replay_transition_limit": maximum_replay_transitions,
        "static_replay_tape_forwarded_external_calls": list(
            forwarding_port.forwarded_request_ids
        ),
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
