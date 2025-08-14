
from dataclasses import dataclass

@dataclass
class EngineParams:
    dt: float = 1e-2
    substeps: int = 20
    iterations: int = 100
    damping: float = 0.69
    dimension: int = 3

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
    # Chunking knobs for ``build_self_contacts_spatial_hash``.  ``max_vox_entries``
    # and ``vbatch`` are shrunk automatically when the estimated temporary array
    # footprint exceeds ``contact_ram_limit``.
    contact_max_vox_entries: int = 8_000_000
    contact_vbatch: int = 250_000
    contact_ram_limit: int | None = 1024 * 1024 * 1024 * 10  # bytes; ``None`` = no cap

    # Thin bath (Z is 1 voxel thick in the demo)
    bath_min = (-1.0, -1.0, -1.0)
    bath_max = (1.0, 1.0, 1.00)

    gravity: tuple = (0.0, 0.0, 0.0)  # not used in demo
