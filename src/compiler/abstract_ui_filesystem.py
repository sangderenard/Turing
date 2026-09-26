"""Human artifacts and a backend-neutral filesystem/ownership graph.

Filesystem containment, conceptual ownership, and spatial placement are
deliberately orthogonal.  A source file may live in one path, belong to a
class, and be placed loose in a remote courtyard without changing identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


ABSTRACT_UI_FILESYSTEM_VERSION = "abstract-ui-filesystem-v0"
HUMAN_ARTIFACT_KINDS = ("annotation", "readme", "scratch", "source", "test")


@dataclass(frozen=True, slots=True)
class ArtifactAttachment:
    owner: str
    state: str = "loose"
    settle_time: float = 0.0
    connection_radius: float = 0.42
    maximum_connection_speed: float = 0.12
    required_settle_time: float = 0.65
    weld_policy: str = "close-slow-settle"

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("artifact attachment requires an owner")
        if self.state not in {"loose", "settling", "welded"}:
            raise ValueError(f"unknown artifact attachment state {self.state!r}")
        if min(self.connection_radius, self.maximum_connection_speed,
               self.required_settle_time) < 0:
            raise ValueError("artifact attachment thresholds cannot be negative")

    def to_data(self) -> dict[str, Any]:
        return {
            "owner": self.owner, "state": self.state,
            "settle_time": self.settle_time,
            "connection_radius": self.connection_radius,
            "maximum_connection_speed": self.maximum_connection_speed,
            "required_settle_time": self.required_settle_time,
            "weld_policy": self.weld_policy,
        }


def advance_attachment(
    attachment: ArtifactAttachment,
    *,
    distance: float,
    relative_speed: float,
    dt: float,
) -> ArtifactAttachment:
    """Advance the close-and-slow welding state without moving either body."""

    if min(distance, relative_speed, dt) < 0:
        raise ValueError("attachment observations and dt cannot be negative")
    if attachment.state == "welded":
        return attachment
    close = distance <= attachment.connection_radius
    slow = relative_speed <= attachment.maximum_connection_speed
    if not (close and slow):
        return replace(attachment, state="loose", settle_time=0.0)
    settled = attachment.settle_time + dt
    return replace(
        attachment,
        state=(
            "welded" if settled >= attachment.required_settle_time
            else "settling"
        ),
        settle_time=settled,
    )


@dataclass(frozen=True, slots=True)
class HumanArtifact:
    identity: str
    kind: str
    name: str
    path: str
    owner: str
    placed_in: str
    attachment: ArtifactAttachment
    media_type: str = "text/plain"
    content_authority: str = "filesystem-node"
    palette_role: str = "artifact-source"

    def __post_init__(self) -> None:
        if self.kind not in HUMAN_ARTIFACT_KINDS:
            raise ValueError(f"unknown human artifact kind {self.kind!r}")
        if self.attachment.owner != self.owner:
            raise ValueError("artifact and attachment owner must agree")
        if not self.identity or not self.name or not self.owner or not self.placed_in:
            raise ValueError("artifact identity, name, owner, and placement are required")
        canonical_path(self.path)

    def to_data(self) -> dict[str, Any]:
        welded = self.attachment.state == "welded"
        return {
            "identity": self.identity, "kind": self.kind, "name": self.name,
            "path": canonical_path(self.path), "owner": self.owner,
            "placed_in": self.placed_in, "media_type": self.media_type,
            "content_authority": self.content_authority,
            "palette_role": self.palette_role,
            "attachment": self.attachment.to_data(),
            "physics": {
                "body": "compound-child" if welded else "dynamic",
                "collider": "owner-compound" if welded else "solid-box",
                "embedded": True,
                "mass": 0.08,
                "linear_drag": 2.4,
                "welded": welded,
            },
            "capabilities": [
                "inspect", "read", "edit", "move", "attach", "detach",
                "reveal-owner", "reveal-filesystem-location",
            ],
            "dependencies": [
                {"relationship": "owned-by", "target": self.owner},
                {"relationship": "placed-in", "target": self.placed_in},
            ],
        }


@dataclass(frozen=True, slots=True)
class FileSystemNode:
    identity: str
    path: str
    kind: str
    parent: str | None
    artifact: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "path": self.path, "name": (
                "/" if self.path == "/" else PurePosixPath(self.path).name
            ),
            "kind": self.kind, "parent": self.parent, "artifact": self.artifact,
            "dependencies": ([] if self.parent is None else [
                {"relationship": "contained-by", "target": self.parent},
            ]),
        }


@dataclass(frozen=True, slots=True)
class FileSystemGraph:
    identity: str
    root: str
    nodes: tuple[FileSystemNode, ...]
    artifacts: tuple[HumanArtifact, ...]

    def __post_init__(self) -> None:
        identities = [node.identity for node in self.nodes]
        paths = [node.path for node in self.nodes]
        if len(identities) != len(set(identities)) or len(paths) != len(set(paths)):
            raise ValueError("filesystem node identities and paths must be unique")
        known = set(identities)
        if self.root not in known:
            raise ValueError("filesystem root node is missing")
        if any(node.parent is not None and node.parent not in known for node in self.nodes):
            raise ValueError("filesystem node has a missing parent")
        artifact_ids = {artifact.identity for artifact in self.artifacts}
        referenced = {node.artifact for node in self.nodes if node.artifact is not None}
        if artifact_ids != referenced:
            raise ValueError("every artifact requires exactly one filesystem file node")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_FILESYSTEM_VERSION,
            "identity": self.identity, "root": self.root,
            "path_semantics": "canonical-posix-logical-path",
            "nodes": [node.to_data() for node in self.nodes],
            "artifacts": [artifact.to_data() for artifact in self.artifacts],
            "ownership_edges": [
                {"source": artifact.identity, "relationship": "owned-by",
                 "target": artifact.owner}
                for artifact in self.artifacts
            ],
            "placement_edges": [
                {"source": artifact.identity, "relationship": "placed-in",
                 "target": artifact.placed_in}
                for artifact in self.artifacts
            ],
            "backend_contracts": {
                "internal": {
                    "storage": "revisioned-virtual-file-graph",
                    "path_authority": "filesystem-node",
                    "content": "inline-or-content-addressed-blob",
                    "structure_binding": "artifact-ownership-edges",
                },
                "native-c": {
                    "storage": "host-filesystem-adapter",
                    "operations": ["open", "read", "write", "stat", "rename"],
                    "boundary": "explicit-root-capability-no-ambient-path-access",
                    "structure_binding": "translation-unit-and-symbol-table-edges",
                },
                "web": {
                    "storage": "virtual-manifest-with-opfs-or-indexeddb-adapter",
                    "operations": ["read", "write", "list", "move", "snapshot"],
                    "boundary": "browser-origin-sandbox-no-host-filesystem-claim",
                    "structure_binding": "module-url-and-bundle-manifest-edges",
                },
                "wasm": {
                    "storage": "host-imported-filesystem-abi",
                    "identity": "dense-node-id-with-reversible-path-table",
                    "structure_binding": "host-import-and-source-map-edges",
                },
            },
        }


def canonical_path(path: str) -> str:
    raw = str(path).replace("\\", "/")
    value = PurePosixPath("/" + raw.lstrip("/"))
    if ".." in value.parts:
        raise ValueError("filesystem paths cannot escape the logical root")
    return str(value)


def _filesystem_nodes(
    identity: str,
    artifacts: Iterable[HumanArtifact],
) -> tuple[FileSystemNode, ...]:
    root_identity = f"{identity}/path:/"
    nodes = [FileSystemNode(root_identity, "/", "directory", None)]
    by_path = {"/": root_identity}
    for artifact in artifacts:
        path = canonical_path(artifact.path)
        parent_path = "/"
        for part in PurePosixPath(path).parts[1:-1]:
            child_path = canonical_path(f"{parent_path}/{part}")
            if child_path not in by_path:
                node_identity = f"{identity}/path:{child_path}"
                nodes.append(FileSystemNode(
                    node_identity, child_path, "directory", by_path[parent_path],
                ))
                by_path[child_path] = node_identity
            parent_path = child_path
        nodes.append(FileSystemNode(
            f"{identity}/path:{path}", path, "file", by_path[parent_path],
            artifact.identity,
        ))
        by_path[path] = nodes[-1].identity
    return tuple(nodes)


def filesystem_graph_for_world(
    world: Mapping[str, Any],
    *,
    actor: str | None = None,
) -> FileSystemGraph:
    """Create the reference human-artifact set for an introspective world."""

    world_identity = str(world["identity"])
    buildings = [
        building
        for region in world.get("regions", ())
        for building in region.get("buildings", ())
    ]
    if not buildings:
        raise ValueError("filesystem projection requires an owning building")
    owner = str(buildings[0]["identity"])
    module = str(buildings[0].get("module") or buildings[0].get("name") or "program")
    module_path = module.replace(".", "/")
    stem = module.rsplit(".", 1)[-1]
    placement = f"{world_identity}/representation:global"
    definitions = (
        ("source", f"{module_path}.py", f"{stem}.py", owner, "artifact-source"),
        ("test", f"tests/test_{stem}.py", f"test_{stem}.py", owner, "artifact-test"),
        ("readme", "README.md", "README.md", world_identity, "artifact-readme"),
        ("annotation", f".abstractui/annotations/{stem}.md", f"{stem} notes", owner,
         "artifact-annotation"),
        ("scratch", f".abstractui/scratch/{stem}.txt", f"{stem} scratch",
         actor or world_identity, "artifact-scratch"),
    )
    artifacts = tuple(
        HumanArtifact(
            f"{world_identity}/artifacts/{kind}:{index}", kind, name, path,
            artifact_owner, placement, ArtifactAttachment(artifact_owner),
            "text/markdown" if path.endswith(".md") else "text/plain",
            palette_role=palette_role,
        )
        for index, (kind, path, name, artifact_owner, palette_role)
        in enumerate(definitions)
    )
    graph_identity = f"{world_identity}/filesystem"
    nodes = _filesystem_nodes(graph_identity, artifacts)
    return FileSystemGraph(
        graph_identity, f"{graph_identity}/path:/", nodes, artifacts,
    )


def artifact_geometry_boxes(
    graph: FileSystemGraph,
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Place loose artifact tokens in global space without changing ownership."""

    envelope = next(box for box in geometry["boxes"] if box["kind"] == "world-envelope")
    courtyard = next(box for box in geometry["boxes"] if box["kind"] == "courtyard")
    start_x = courtyard["center"][0] - courtyard["half_extent"][0]
    z = min(
        envelope["center"][1] + envelope["half_extent"][1] - 1.5,
        courtyard["center"][1] + courtyard["half_extent"][1] + 2.0,
    )
    boxes = []
    for index, artifact in enumerate(graph.artifacts):
        data = artifact.to_data()
        boxes.append({
            "identity": artifact.identity, "kind": "human-artifact",
            "artifact_kind": artifact.kind, "label": artifact.name,
            "parent_identity": artifact.owner,
            "spatial_container": artifact.placed_in,
            "hierarchy_depth": 1, "center": [start_x + index * 0.72, z],
            "half_extent": [0.24, 0.32], "height": 0.42,
            "floor_height": 0.0, "wall_thickness": 0.0, "radius": 3.0,
            "palette_role": artifact.palette_role,
            "wall_palette_role": artifact.palette_role,
            "geometry_mode": "solid", "openings": [],
            "artifact": data, "attachment": data["attachment"],
            "physics": data["physics"],
        })
    return tuple(boxes)


__all__ = [
    "ABSTRACT_UI_FILESYSTEM_VERSION", "HUMAN_ARTIFACT_KINDS",
    "ArtifactAttachment", "FileSystemGraph", "FileSystemNode", "HumanArtifact",
    "advance_attachment", "artifact_geometry_boxes", "canonical_path",
    "filesystem_graph_for_world",
]
