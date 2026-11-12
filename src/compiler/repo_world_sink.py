"""User-space sink for repository world-map records and dream work tables.

The scan (``repo_world_scan``) and the work table (``dream_document``) return
records in memory; what happens to them is the call's own declaration.  This
module is the declaration "store in user space": it seals a bounded bundle of
those records into the owner's **account workspace** through the service's
existing store, ``examples/meowdoku/workspace_accounting.Store`` (per-account
leases, quotas, sealed review snapshots, admin review).  Nothing here writes a
file, opens a database of its own, or decides who the actor is: the caller
hands in a ``Store`` and an ``actor_id`` that *is* a ``users.id`` row of the
service accounts database, the only identity authority.

The store's review bundle is bounded (``MAX_SNAPSHOT_BYTES`` = 1 MiB, 128
files).  A whole scan of a real repository is tens of megabytes, so the
bundle is the *personal* part of the record set, not the whole record set:

    summary.json      commit, origin, counts, the languages table, notes
    claims.json       every claimed room (the stand-ins' ownership)
    work_table.json   a dream document's stand-ins and satisfaction tree
    scan.json         the full record set, ONLY when the call asks for it
                      (``full=True``) and it fits; the store refuses otherwise

The bundle's ``base_revision`` is the sha256 of the canonical JSON of what is
being stored, so the lease names exactly the record it holds; the git commit
is inside ``summary.json``.  Ticket semantics are the store's: an identical
bundle resubmitted returns the same ticket.

No CLI: the actor is a session the service authenticates; see
``workspace_accounting.SessionAccess``.  This module is for the service (or a
test with a fixture ``Store``) to call.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .repo_world_scan import RepoWorldScan


def canonical_bytes(value: Any) -> bytes:
    """One byte form per value, so digests and resubmissions agree."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def scan_bundle(scan: RepoWorldScan, *, full: bool = False) -> dict[str, bytes]:
    """The personal part of a scan as a ``{path: bytes}`` bundle for the store."""

    data = scan.to_data()
    summary = {
        "schema": "repo-world-sink-summary-v0",
        "scan_schema": data["schema"],
        "identity": data["identity"],
        "commit": data["commit"],
        "origin": data["origin"],
        "counts": {
            "regions": len(scan.regions), "rooms": len(scan.rooms),
            "portal_pairs": len(scan.portals), "unresolved_references": len(scan.unresolved),
            "claimed_rooms": len(scan.claims),
        },
        "languages": data["languages"],
        "notes": data["notes"],
    }
    bundle = {
        "summary.json": canonical_bytes(summary),
        "claims.json": canonical_bytes(data["claims"]),
    }
    if full:
        bundle["scan.json"] = canonical_bytes(data)
    return bundle


def work_table_bundle(document: Any, *, source: str | None = None) -> dict[str, bytes]:
    """A dream document's work table (stand-ins, authors, satisfaction tree)."""

    table = document.work_table().to_data()
    if source is not None:
        table = {**table, "source": source}
    return {"work_table.json": canonical_bytes(table)}


def bundle_revision(bundle: Mapping[str, bytes]) -> str:
    """sha256 over the bundle's paths and bytes: the lease's ``base_revision``."""

    digest = sha256()
    for path in sorted(bundle):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bundle[path])
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SinkReceipt:
    """What the store gave back: where the bundle lives and under which ticket."""

    workspace_id: str
    ticket_id: str
    base_revision: str
    snapshot_digest: str
    snapshot_bytes: int
    files: tuple[str, ...]

    def to_data(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id, "ticket_id": self.ticket_id,
            "base_revision": self.base_revision, "snapshot_digest": self.snapshot_digest,
            "snapshot_bytes": self.snapshot_bytes, "files": list(self.files),
        }


def submit(store: Any, actor_id: int, bundle: Mapping[str, bytes], *,
           workspace_id: str | None = None) -> SinkReceipt:
    """Seal ``bundle`` into the actor's workspace and return the store's receipt.

    With no ``workspace_id`` a new lease is reserved for this bundle, sized
    from the bundle (the store charges the encoded payload and admits any
    excess against the actor's quota itself).  With one, the bundle is sealed
    into that existing lease at its current generation.  Every refusal is the
    store's (``Refused``, ``NotFound``, ``ValueError``) and is not caught here:
    an unknown actor, an exhausted quota, an over-size bundle or an unsafe path
    all stop before anything is stored.
    """

    bundle = dict(bundle)
    if not bundle:
        raise ValueError("an empty bundle has nothing to store")
    revision = bundle_revision(bundle)
    if workspace_id is None:
        # The store keeps base64 plus a JSON envelope per file; 4/3 of the
        # bytes plus a per-file allowance covers it, and seal_and_submit admits
        # any remainder against the quota before writing.
        estimate = sum(len(raw) for raw in bundle.values()) * 4 // 3 + 512 * len(bundle) + 512
        workspace_id = store.reserve(actor_id, estimate, base_revision=revision)
    lease = store.get(actor_id, workspace_id)
    ticket_id = store.seal_and_submit(
        actor_id, workspace_id, bundle, expected_generation=lease["generation"],
    )
    ticket = store.get_ticket(actor_id, ticket_id)
    return SinkReceipt(
        workspace_id, ticket_id, lease["base_revision"], ticket["snapshot_digest"],
        ticket["snapshot_bytes"], tuple(sorted(bundle)),
    )


def read_bundle(store: Any, actor_id: int, ticket_id: str) -> dict[str, bytes]:
    """The owner's bundle back from the store, path -> bytes, integrity-checked by the store."""

    import base64

    payload = json.loads(store.read_snapshot(actor_id, ticket_id))
    return {row["path"]: base64.b64decode(row["base64"]) for row in payload["files"]}


__all__ = [
    "SinkReceipt", "bundle_revision", "canonical_bytes", "read_bundle",
    "scan_bundle", "submit", "work_table_bundle",
]
