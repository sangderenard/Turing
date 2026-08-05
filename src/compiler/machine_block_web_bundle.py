"""Package one provenance-bound recompiled machine block for a browser shell."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from .machine_block_recompiler import (
    JOURNAL_STRIDE,
    MACHINE_BLOCK_JOURNAL_SCHEMA,
    MACHINE_BLOCK_STATE_SCHEMA,
)


@dataclass(frozen=True, slots=True)
class MachineBlockWebBundle:
    """Executable assets and bootstrap descriptor for one lowered AMD64 block."""

    assets: Mapping[str, bytes]
    descriptor: Mapping[str, Any]
    plan: Mapping[str, Any]


def build_machine_block_web_bundle(
    artifact,
    state,
    *,
    subject_sha256: str,
    root: str = "machine/recompiled-entry",
) -> MachineBlockWebBundle:
    """Return browser-loadable Wasm/state/journal provenance artifacts."""
    normalized = PurePosixPath(root).as_posix().strip("/")
    if not normalized or normalized.startswith("../") or "/../" in normalized:
        raise ValueError("machine-block web root escapes its bundle")
    if not artifact.witnesses:
        raise ValueError("browser machine block requires at least one lowered witness")
    witnesses = [{
        "operation_index": int(item.operation_index),
        "address": int(item.address),
        "semantic": str(item.semantic),
        "semantic_id": int(item.semantic_id),
        "encoded_hex": bytes(item.encoded).hex(),
        "encoded_sha256": str(item.encoded_digest),
        "journal_offset": int(item.journal_offset),
        "possible_next_addresses": [int(value) for value in item.possible_next_addresses],
        "expected_stack_effect": [int(value) for value in item.expected_stack_effect],
    } for item in artifact.witnesses]
    shortfalls = [{
        "operation_index": int(item.operation_index),
        "address": int(item.address),
        "semantic": str(item.semantic),
        "reason": str(item.reason),
    } for item in artifact.shortfalls]
    plan = {
        "schema": "turing.machine-block-web-bundle.v1",
        "subject_sha256": str(subject_sha256),
        "entry_address": int(artifact.entry_address),
        "block_digest": str(artifact.block_digest),
        "complete": bool(artifact.complete),
        "continuation_address": int(artifact.continuation_address),
        "possible_continuations": [int(value) for value in artifact.possible_continuations],
        "covered_operation_count": int(artifact.covered_operation_count),
        "state_schema": MACHINE_BLOCK_STATE_SCHEMA,
        "journal_schema": MACHINE_BLOCK_JOURNAL_SCHEMA,
        "journal_stride": JOURNAL_STRIDE,
        "journal_bytes": int(artifact.covered_operation_count) * JOURNAL_STRIDE,
        "guest_memory_base": int(artifact.guest_memory_base),
        "guest_memory_size": int(artifact.guest_memory_size),
        "state_abi": dict(artifact.state_abi),
        "specialization_guard": dict(artifact.specialization_guard),
        "witnesses": witnesses,
        "shortfalls": shortfalls,
    }
    paths = {
        "module": f"{normalized}/block.wasm",
        "wat": f"{normalized}/block.wat",
        "state": f"{normalized}/initial-state.bin",
        "guest": f"{normalized}/guest-window.bin",
        "plan": f"{normalized}/dispatch-plan.json",
    }
    first = witnesses[0]
    descriptor = {
        "schema": "turing.machine-block-browser-bootstrap.v1",
        **paths,
        "journal_bytes": plan["journal_bytes"],
        "guest_memory_base": plan["guest_memory_base"],
        "entry_address": plan["entry_address"],
        "block_digest": plan["block_digest"],
        "expected_first_witness": {
            "address": first["address"],
            "semantic_id": first["semantic_id"],
            "digest_prefix": first["encoded_sha256"][:16],
        },
    }
    assets = {
        paths["module"]: bytes(artifact.binary),
        paths["wat"]: str(artifact.wat).encode("utf-8"),
        paths["state"]: bytes(artifact.pack_state(state)),
        paths["guest"]: bytes(artifact.pack_guest_memory(state)),
        paths["plan"]: json.dumps(plan, indent=2, default=str).encode("utf-8"),
    }
    return MachineBlockWebBundle(assets=assets, descriptor=descriptor, plan=plan)


__all__ = ["MachineBlockWebBundle", "build_machine_block_web_bundle"]
