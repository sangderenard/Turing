from dataclasses import dataclass, field
from ..engine_api import DtCompatibleEngine
from ..dt_scaler import Metrics
from typing import Hashable, Tuple, Optional, List, Dict
import numpy as np

class MagicCenterOfMass:
    pass
COM: MagicCenterOfMass = MagicCenterOfMass()  # Placeholder for center of mass identifier


@dataclass
class WorldAnchor:
    position: Tuple[float, float]
    massless: bool = True  # Always treated as infinite mass in force calculations

@dataclass
class FreeVertexAnchor:
    position: Tuple[float, float]
    massless: bool = True  # Can be influenced by attached rigid body
    attached_body: Optional[int] = None  # Index or ID of the rigid body, if any


@dataclass
class WorldObjectLink:
    world_anchor: WorldAnchor
    object_anchor: tuple  # (world_anchor_index, vertex_set_identifier, set_index, mass)
    link_type: str  # 'rope', 'steel_beam', 'spring', 'gas_dampner', 'lever_arm'
    properties: Optional[dict] = field(default_factory=dict)



class RigidBodyEngine(DtCompatibleEngine):
    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out['nodes'] = self.nodes.copy()
        out['velocities'] = self.velocities.copy()
        out['masses'] = self.masses.copy()
        return out

    def __init__(self, links: List[WorldObjectLink], state_table, rigid_body_groups: List[dict]):
        """
        links: list of WorldObjectLink
        state_table: shared StateTable instance
        rigid_body_groups: list of dicts, each with keys:
            - 'label': unique group label
            - 'vertices': set or list of vertex indices or ids
            - 'edges': (optional) dict of {color: set(edge_ids)}
            - 'faces': (optional) set/list of face ids
            - ... (any extra features)
        """
        self.links = links
        self.object_anchors = [link.object_anchor for link in links]
        self.nodes = np.array([link.world_anchor.position for link in links], dtype=float)
        self.masses = np.array([oa[3] for oa in self.object_anchors], dtype=float)
        self.velocities = np.zeros_like(self.nodes)
        self.forces = np.zeros_like(self.nodes)
        self._constraint_accum = 0.0
        # Explicit group registration with identity assignment
        import uuid
        self.rigid_body_groups = []
        for group in rigid_body_groups:
            label = group['label']
            vertices = set(group['vertices'])
            edges = group.get('edges', None)
            faces = group.get('faces', None)
            # Assign UUID identity to group if not present
            group_uuid = group.get('uuid', str(uuid.uuid4()))
            # Register group identity with COM marker for position
            if hasattr(state_table, 'register_identity'):
                from dataclasses import dataclass, field
from ..engine_api import DtCompatibleEngine
from typing import Hashable, Tuple, Optional, List, Dict
import numpy as np

class MagicCenterOfMass:
    pass
COM: MagicCenterOfMass = MagicCenterOfMass()  # Placeholder for center of mass identifier


@dataclass
class WorldAnchor:
    position: Tuple[float, float]
    massless: bool = True  # Always treated as infinite mass in force calculations

@dataclass
class FreeVertexAnchor:
    position: Tuple[float, float]
    massless: bool = True  # Can be influenced by attached rigid body
    attached_body: Optional[int] = None  # Index or ID of the rigid body, if any


@dataclass
class WorldObjectLink:
    world_anchor: WorldAnchor
    object_anchor: tuple  # (world_anchor_index, vertex_set_identifier, set_index, mass)
    link_type: str  # 'rope', 'steel_beam', 'spring', 'gas_dampner', 'lever_arm'
    properties: Optional[dict] = field(default_factory=dict)



class RigidBodyEngine(DtCompatibleEngine):
    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out['nodes'] = self.nodes.copy()
        out['velocities'] = self.velocities.copy()
        out['masses'] = self.masses.copy()
        return out

    def __init__(self, links: List[WorldObjectLink], state_table, rigid_body_groups: List[dict]):
        """
        links: list of WorldObjectLink
        state_table: shared StateTable instance
        rigid_body_groups: list of dicts, each with keys:
            - 'label': unique group label
            - 'vertices': set or list of vertex indices or ids
            - 'edges': (optional) dict of {color: set(edge_ids)}
            - 'faces': (optional) set/list of face ids
            - ... (any extra features)
        """
        self.links = links
        self.object_anchors = [link.object_anchor for link in links]
        self.nodes = np.array([link.world_anchor.position for link in links], dtype=float)
        self.masses = np.array([oa[3] for oa in self.object_anchors], dtype=float)
        self.velocities = np.zeros_like(self.nodes)
        self.forces = np.zeros_like(self.nodes)
        self._constraint_accum = 0.0
        # Explicit group registration with identity assignment
        import uuid
        self.rigid_body_groups = []
        for group in rigid_body_groups:
            label = group['label']
            vertices = set(group['vertices'])
            edges = group.get('edges', None)
            faces = group.get('faces', None)
            # Assign UUID identity to group if not present
            group_uuid = group.get('uuid', str(uuid.uuid4()))
            # Register group identity with COM marker for position
            if hasattr(state_table, 'register_identity'):
                state_table.register_identity(pos='COM', mass=0.0, uuid_str=group_uuid)
            # Assign UUIDs to edges and faces if not already present, and register their identity
            edge_identities = {}
            if edges:
                for color, edge_set in edges.items():
                    for edge in edge_set:
                        if isinstance(edge, tuple) and len(edge) == 2:
                            edge_uuid = f"edge_{label}_{edge[0]}_{edge[1]}"
                            # COM marker: edge position is the COM of its two vertices
                            edge_pos_marker = ('COM', edge)
                        else:
                            edge_uuid = f"edge_{label}_{str(edge)}"
                            edge_pos_marker = 'COM'  # fallback
                        edge_identities[edge] = edge_uuid
                        if hasattr(state_table, 'register_identity'):
                            state_table.register_identity(pos=edge_pos_marker, mass=0.0, uuid_str=edge_uuid)
            face_identities = {}
            if faces:
                for face in faces:
                    face_uuid = f"face_{label}_{str(face)}"
                    # COM marker: face position is the COM of its vertices (assume face is a tuple/list of vertex indices)
                    if isinstance(face, (tuple, list)):
                        face_pos_marker = ('COM', tuple(face))
                    else:
                        face_pos_marker = 'COM'
                    face_identities[face] = face_uuid
                    if hasattr(state_table, 'register_identity'):
                        state_table.register_identity(pos=face_pos_marker, mass=0.0, uuid_str=face_uuid)
            # Register in state_table
            if hasattr(state_table, 'register_group'):
                state_table.register_group(
                    group_label=label,
                    vertices=vertices,
                    edges=edges,
                    faces=faces
                )
            self.rigid_body_groups.append(label)
    def _lookup_vertex_group(
        self, *, state_table, vertex_set_identifier, set_index
    ):
        
        """Look up a vertex group (e.g., all vertices for an object) from the state table using group accessors only.
        Uses state_table.get_group_vertices and state_table.get_identity for positions and masses."""
        if state_table is None:
            raise ValueError("RigidBodyEngine requires explicit state_table for COM/group lookups.")
        vertex_uuids = list(state_table.get_group_vertices(vertex_set_identifier))
        if not vertex_uuids:
            raise ValueError(f"No vertices found for group '{vertex_set_identifier}' in state_table.")
        # Get positions and masses from identity_registry
        pos = np.asarray([state_table.get_identity(uuid)['pos'] for uuid in vertex_uuids])
        mass = np.asarray([state_table.get_identity(uuid)['mass'] for uuid in vertex_uuids])
        if set_index is not None and type(set_index) is not MagicCenterOfMass:
            pos = pos[set_index]
            mass = mass[set_index]
        elif type(set_index) is MagicCenterOfMass:
            # Compute center of mass
            com = np.sum(pos * mass[:, None], axis=0) / np.sum(mass) if np.sum(mass) > 0 else np.zeros(3)
            
            mass = np.array([np.sum(mass)])
        return com, mass

    def apply_forces(self, state_table=None):
        self.forces[:] = 0.0
        for i, link in enumerate(self.links):
            # object_anchor: (world_anchor_index, vertex_set_identifier, set_index, mass)
            world_anchor_index, vertex_set_identifier, set_index, mass = link.object_anchor
            anchor_pos = np.array(link.world_anchor.position, dtype=float)
            # If anchor is COM, look up group and compute center of mass
            if vertex_set_identifier is COM:
                # Use the world_anchor_index to find the object identifier
                # For now, assume set_index is the object id/name
                object_id = set_index
                
                pos, mass_arr = self._lookup_vertex_group(state_table=state_table, vertex_set_identifier=object_id, set_index=None)
                total_mass = np.sum(mass_arr)
                if total_mass < 1e-12:
                    com = np.mean(pos, axis=0)
                else:
                    com = np.sum(pos * mass_arr[:, None], axis=0) / total_mass
                node_pos = com
                node_mass = total_mass
            else:
                # Otherwise, look up the specific vertex (or group)
                pos, mass_arr = self._lookup_vertex_group(state_table=state_table, vertex_set_identifier=vertex_set_identifier, set_index=set_index)
                node_pos = pos
                node_mass = mass_arr
            delta = node_pos - anchor_pos
            dist = np.linalg.norm(delta)
            direction = delta / dist if dist > 1e-8 else np.zeros_like(delta)
            # Rope: only tensile, no compression
            if link.link_type == 'rope':
                max_length = link.properties.get('max_length', 1.0)
                if dist > max_length:
                    k = link.properties.get('k', 100.0)
                    force = -k * (dist - max_length) * direction
                    self.forces[i] += force
            # Steel beam: rigid, or nearly rigid (very high k)
            elif link.link_type == 'steel_beam':
                length = link.properties.get('length', 1.0)
                k = link.properties.get('k', 10000.0)
                force = -k * (dist - length) * direction
                self.forces[i] += force
            # Spring: classic Hooke's law
            elif link.link_type == 'spring':
                rest_length = link.properties.get('rest_length', 1.0)
                k = link.properties.get('k', 50.0)
                force = -k * (dist - rest_length) * direction
                self.forces[i] += force
            # Gas dampner: velocity-based damping
            elif link.link_type == 'gas_dampner':
                damping = link.properties.get('damping', 5.0)
                v = self.velocities[i]
                force = -damping * v
                self.forces[i] += force
            # Lever arm: like a steel beam, but no rotation allowed (fix angle)
            elif link.link_type == 'lever_arm':
                length = link.properties.get('length', 1.0)
                k = link.properties.get('k', 10000.0)
                force = -k * (dist - length) * direction
                self.forces[i] += force
                # TODO: add angular constraint (no rotation) if needed

    def step(self, dt: float, state=None, state_table=None):
        # Optionally update internal state from state dict
        if isinstance(state, dict):
            if 'nodes' in state:
                self.nodes = np.asarray(state['nodes'], dtype=float)
            if 'velocities' in state:
                self.velocities = np.asarray(state['velocities'], dtype=float)
            if 'masses' in state:
                self.masses = np.asarray(state['masses'], dtype=float)
        self.apply_forces(state_table=state_table)
        max_vel = float(np.max(np.linalg.norm(self.velocities, axis=1))) if len(self.velocities) else 0.0
        err = float(np.max(np.linalg.norm(self.forces, axis=1))) if len(self.forces) else 0.0
        self._constraint_accum += err * dt
        dt_limit = 0.5 / (max_vel + 1e-9) if max_vel > 0 else None
        metrics = Metrics(max_vel=max_vel, max_flux=0.0, div_inf=self._constraint_accum, mass_err=0.0, dt_limit=dt_limit)
        return True, metrics, self.get_state()


# Example usage/defaults
if __name__ == "__main__":
    anchors = [WorldAnchor(position=(0.0, 0.0)), WorldAnchor(position=(2.0, 0.0))]
    nodes = [(1.0, 1.0), (2.0, 2.0)]
    masses = [1.0, 1.0]
    links = [
        WorldObjectLink(anchor=anchors[0], object_vertex=0, link_type='rope', properties={'max_length': 1.5, 'k': 200.0}),
        WorldObjectLink(anchor=anchors[1], object_vertex=1, link_type='steel_beam', properties={'length': 1.0, 'k': 10000.0}),
    ]
    engine = RigidBodyEngine(nodes, masses, links)
    for _ in range(10):
        engine.step(0.01)
        print(engine.nodes)

