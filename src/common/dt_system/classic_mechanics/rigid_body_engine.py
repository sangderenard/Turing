from dataclasses import dataclass, field

from typing import Tuple, Optional, List, Dict
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



class RigidBodyEngine:
    def __init__(self, links: List[WorldObjectLink]):
        self.links = links
        self.object_anchors = [link.object_anchor for link in links]
        # Placeholder arrays; real positions/velocities will be looked up via state_table
        self.nodes = np.array([link.world_anchor.position for link in links], dtype=float)
        self.masses = np.array([oa[3] for oa in self.object_anchors], dtype=float)
        self.velocities = np.zeros_like(self.nodes)
        self.forces = np.zeros_like(self.nodes)

    def _lookup_vertex_group(self, vertex_set_identifier, state_table, set_index=None):
        """Look up a vertex group (e.g., all vertices for an object) from the state table."""
        # Convention: state_table.get(scope, name, field_name)
        # For classic mechanics, use ('object', vertex_set_identifier, 'pos')
        if state_table is None:
            raise ValueError("RigidBodyEngine requires explicit state_table for COM/group lookups.")
        pos = state_table.get('object', vertex_set_identifier, 'pos')
        mass = state_table.get('object', vertex_set_identifier, 'mass')
        if pos is None or mass is None:
            raise ValueError(f"No position/mass found for object '{vertex_set_identifier}' in state_table.")
        pos = np.asarray(pos)
        mass = np.asarray(mass)
        if set_index is not None:
            # Optionally select a subset (e.g., a specific group)
            pos = pos[set_index]
            mass = mass[set_index]
        return pos, mass

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
                pos, mass_arr = self._lookup_vertex_group(object_id, state_table)
                total_mass = np.sum(mass_arr)
                if total_mass < 1e-12:
                    com = np.mean(pos, axis=0)
                else:
                    com = np.sum(pos * mass_arr[:, None], axis=0) / total_mass
                node_pos = com
                node_mass = total_mass
            else:
                # Otherwise, look up the specific vertex (or group)
                pos, mass_arr = self._lookup_vertex_group(vertex_set_identifier, state_table, set_index)
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

    def step(self, dt: float, state_table=None):
        self.apply_forces(state_table=state_table)
        # Integrate motion for free vertices (if needed)
        for i, mass in enumerate(self.masses):
            if mass > 0:
                acc = self.forces[i] / mass
                self.velocities[i] += acc * dt
                self.nodes[i] += self.velocities[i] * dt
            # Anchored nodes (infinite mass) do not move

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
