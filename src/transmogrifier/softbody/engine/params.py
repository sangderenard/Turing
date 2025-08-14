
from dataclasses import dataclass

@dataclass
class EngineParams:
    dt: float = 0.01
    substeps: int = 6
    iterations: int = 10
    damping: float = 0.996

    # XPBD compliances (1/k)
    stretch_compliance: float = 1e-6
    volume_compliance: float  = 1e-6
    bending_compliance: float = 2e-5
    contact_compliance: float = 0.0  # 0 = hard non-penetration

    # Broad-phase self-contacts (vertex-triangle within a cell)
    # These can allocate large temporary Python structures (hashes/sets) each step.
    # Demos that don't need intra-mesh collision resolution can disable them to
    # reduce memory churn.
    enable_self_contacts: bool = True
    contact_voxel_size: float = 0.05

    # Thin bath (Z is 1 voxel thick in the demo)
    bath_min = (0.0, 0.0, 0.0)
    bath_max = (1.0, 1.0, 0.02)

    gravity: tuple = (0.0, 0.0, 0.0)  # not used in demo
