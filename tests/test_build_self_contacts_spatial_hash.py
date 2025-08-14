import numpy as np

from src.cells.softbody.engine import build_self_contacts_spatial_hash


def test_build_self_contacts_spatial_hash_simple():
    # Square made of two triangles in the XY plane
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.1],  # vertex above centre
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cell_ids = np.zeros(len(X), dtype=np.int32)

    pairs = build_self_contacts_spatial_hash(X, faces, cell_ids, voxel_size=0.5)
    pairs_set = {tuple(p) for p in pairs}

    # Vertex 4 should be paired with both triangles, while vertices on the
    # surface should not generate contacts due to adjacency filtering.
    assert pairs_set == {(4, 0), (4, 1)}
